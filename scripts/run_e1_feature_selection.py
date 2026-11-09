"""E1 phase A (survey v3 = RQ1): select feature panels only, using precomputed 5x5 outer splits.

Each selection call receives only its manifest's outer-train samples.  The resulting panels are stored separately
from classifier metrics so that Phase B evaluates the approved classifiers without selecting features again.

v3 (QĐ15, 26/10/2026): 13 selectors (11 non-wrapper by default; GA/BPSO via --methods ga bpso, WRAPPER_DATASETS,
--repeats 0 1 2, --k from WRAPPER_K) + fixed knowledge panels (ihc_2, ihc_4, panel_<name> from
survey_v2/panels/published_panels.csv); k grid {1,2,3,4,5,10,25,50}; gene space = intersection with TCGA for GEO
cohorts (loader); k-TSP only at even k (odd k -> n_selected = 0, no genes); emissions tracking OFF by default (Y16).

v3.1.6 (QĐ25, 31/10/2026): --redo-methods M ... deletes the checkpoints of those methods first and recomputes them, while
every other method is taken from the existing checkpoints (restore the previous output as Kaggle Input). The aggregate
files are rebuilt from all checkpoints; config.json records redo_methods and the commit of the reused checkpoints.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import PREFILTER_RULES, drop_genes, load_dataset, prefilter, sha256  # noqa: E402
from survey_v2.e1_splits import validate_repeated_split_manifest  # noqa: E402
from survey_v2.methods import select  # noqa: E402
from survey_v2.panels import all_fixed_panels, panel_indices  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, previous_config, results_dir, start_run, write_config  # noqa: E402

FIXED_PANELS = all_fixed_panels()
DEFAULT_METHODS = list(C.NON_WRAPPER_METHODS) + list(FIXED_PANELS)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS))
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                    help=f"selectors {C.ALL_SELECTORS} and fixed panels {tuple(FIXED_PANELS)}")
    ap.add_argument("--k", nargs="+", type=int, default=list(C.K_GRID))
    ap.add_argument("--repeats", nargs="+", type=int, default=list(range(len(C.REPEAT_SEEDS))))
    ap.add_argument("--prefilter", type=int, default=C.PREFILTER_P0)
    ap.add_argument("--prefilter-rule", choices=PREFILTER_RULES, default="variance",
                    help="variance (mặc định, không dùng nhãn, M14) hoặc ftest (độ nhạy RQ1 QĐ53b: F-test trên train của fold)")
    ap.add_argument("--splits-root", type=Path, default=None,
                    help="root containing <cohort>/repeated_splits.csv; defaults to the cohort data directory")
    ap.add_argument("--out", default="E1_feature_selection")
    ap.add_argument("--pop", type=int, default=C.WRAPPER_POP)
    ap.add_argument("--generations", type=int, default=C.WRAPPER_GENERATIONS)
    ap.add_argument("--exclude-genes", nargs="+", default=None,
                    help="ablation: loại các gen này khỏi không gian gen trước khi lọc/chọn (vd. NKX2-1 TP63 NAPSA KRT5); dùng --out riêng")
    ap.add_argument("--track-emissions", dest="track_emissions", action="store_true",
                    help="CodeCarbon estimate (optional, descriptive only; OFF by default — Y16)")
    ap.add_argument("--redo-methods", nargs="+", default=None,
                    help="v3.1.6: xoá checkpoint của các phương pháp này rồi tính lại; phương pháp khác lấy từ checkpoint đã có")
    ap.set_defaults(track_emissions=False)
    return ap.parse_args(argv)


def split_path(ds, root):
    return (Path(root) / ds.name / "repeated_splits.csv") if root else ds.data_dir / "repeated_splits.csv"


def split_indices(ds, manifest, repeat, fold):
    part = manifest[(manifest.repeat == repeat) & (manifest.fold == fold)]
    index = {sample: i for i, sample in enumerate(ds.sample_ids)}
    if set(part.sample_id) != set(index):
        raise ValueError(f"{ds.name}: manifest samples differ from expression samples")
    expected_patient = dict(zip(ds.sample_ids, ds.groups))
    expected_label = dict(zip(ds.sample_ids, ds.y))
    for row in part[["sample_id", "patient_id", "label"]].itertuples(index=False):
        if expected_patient[row.sample_id] != str(row.patient_id) or expected_label[row.sample_id] != int(row.label):
            raise ValueError(f"{ds.name}: manifest patient_id or label does not match metadata for {row.sample_id}")
    train = np.array([index[s] for s in part.loc[part.role == "train", "sample_id"]], dtype=int)
    test = np.array([index[s] for s in part.loc[part.role == "test", "sample_id"]], dtype=int)
    return train, test


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else float("nan")


def start_emissions_tracker(enabled):
    if not enabled:
        return None
    try:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(project_name="e1_feature_selection", country_iso_code="VNM", save_to_file=False, log_level="error")
        tracker.start()
        return tracker
    except Exception as exc:
        print(f"[emissions] unavailable: {exc}", flush=True)
        return None


def checkpoint_paths(out: Path, repeat: int, fold: int, method: str) -> tuple[Path, Path]:
    """Two complete files are the atomic completion marker for one selection unit."""
    base = out / "checkpoints" / f"repeat_{repeat}_fold_{fold}_{method}"
    return base.with_suffix(".metrics.csv"), base.with_suffix(".genes.csv")


GENE_COLUMNS = ["cohort", "repeat", "fold", "method", "k", "rank", "gene"]


def write_csv_atomic(frame: pd.DataFrame, path: Path, columns=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty and columns is not None:
        frame = pd.DataFrame(columns=columns)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def drop_method_checkpoints(out: Path, methods) -> int:
    """v3.1.6: xoá checkpoint (metrics + genes) của các phương pháp cần tính lại; trả về số tệp đã xoá."""
    import re
    n = 0
    for m in methods:
        pat = re.compile(rf"^repeat_\d+_fold_\d+_{re.escape(m)}\.(metrics|genes)\.csv$")
        for f in (out / "checkpoints").glob(f"repeat_*_fold_*_{m}.*.csv"):
            if pat.match(f.name):                                  # khớp đúng tên phương pháp (không xoá nhầm)
                f.unlink()
                n += 1
    return n


def main(argv=None, script=__file__):
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    if args.redo_methods:
        missing = [m for m in args.redo_methods if m not in args.methods]
        if missing:
            raise SystemExit(f"--redo-methods phải nằm trong --methods: {missing}")
    for name in args.datasets:
        ds, out = load_dataset(DATA_ROOT, name), results_dir(args.out, name)
        prev_cfg = previous_config(out)
        if args.redo_methods:
            n_del = drop_method_checkpoints(out, args.redo_methods)
            print(f"[{name}] --redo-methods {args.redo_methods}: đã xoá {n_del} checkpoint, sẽ tính lại", flush=True)
        if args.exclude_genes:
            ds = drop_genes(ds, args.exclude_genes)
            print(f"[{name}] ablation: loại {ds.notes['excluded_genes']} (không có: {ds.notes['excluded_genes_not_found']})", flush=True)
        path = split_path(ds, args.splits_root)
        manifest = pd.read_csv(path)
        errors = validate_repeated_split_manifest(manifest, n_folds=C.N_FOLDS)
        if errors:
            raise ValueError(f"{name}: invalid repeated_splits.csv: {'; '.join(errors)}")
        process = psutil.Process()
        skipped: set[str] = set()
        runs = manifest[["repeat", "seed", "fold"]].drop_duplicates().sort_values(["repeat", "fold"])
        for run in runs.itertuples(index=False):
            if run.repeat not in args.repeats:
                continue
            tr, _ = split_indices(ds, manifest, run.repeat, run.fold)
            pre = prefilter(ds.X[tr], ds.y[tr], args.prefilter, args.prefilter_rule)
            for method in args.methods:
                metrics_path, genes_path = checkpoint_paths(out, int(run.repeat), int(run.fold), method)
                if metrics_path.is_file() and genes_path.is_file():
                    print(f"[{name}] repeat {run.repeat} fold {run.fold} {method}: checkpoint exists, skip", flush=True)
                    continue
                fold_rows, fold_genes = [], []
                if method in FIXED_PANELS:
                    # panel tri thức / panel công bố: cố định, không phụ thuộc dữ liệu; gen thiếu được ghi lại
                    cols, missing = panel_indices(ds.genes, FIXED_PANELS[method])
                    k_panel = len(FIXED_PANELS[method])
                    fold_rows.append({"cohort": name, "repeat": run.repeat, "seed": run.seed, "fold": run.fold,
                                      "method": method, "k": int(k_panel), "n_train": len(tr), "p0": len(pre),
                                      "n_selected": len(cols), "exact_k": len(missing) == 0, "n_filled": 0,
                                      "selector_param": None, "n_eval": None, "best_fitness": None,
                                      "selection_time_s": 0.0, "cpu_time_s": 0.0, "rss_delta_mb": 0.0,
                                      "emissions_kg_co2": float("nan"), "missing_genes": ";".join(missing)})
                    fold_genes.extend({"cohort": name, "repeat": run.repeat, "fold": run.fold, "method": method,
                                       "k": int(k_panel), "rank": rank, "gene": ds.genes[col]}
                                      for rank, col in enumerate(cols, 1))
                    write_csv_atomic(pd.DataFrame(fold_rows), metrics_path)
                    write_csv_atomic(pd.DataFrame(fold_genes), genes_path, columns=GENE_COLUMNS)
                    continue
                method_id = C.METHOD_ID[method]
                seed = C.seed_for(C.cohort_id(name), int(run.seed), int(run.fold), method_id)
                rss_before, cpu_before, t0 = process.memory_info().rss, time.process_time(), time.perf_counter()
                tracker = start_emissions_tracker(args.track_emissions)
                try:
                    with threadpool_limits(1):
                        selected = select(method, ds.X[tr][:, pre], ds.y[tr], args.k, seed=seed, n_jobs=1,
                                          wrapper_kwargs={"pop": args.pop, "generations": args.generations})
                except ImportError as exc:          # abess không cài được: ghi nhận, không dừng cả đợt
                    print(f"[{name}] {method}: BỎ QUA ({exc})", flush=True)
                    skipped.add(method)
                    if tracker:
                        tracker.stop()
                    continue
                emissions_kg = tracker.stop() if tracker else float("nan")
                elapsed, cpu_s = time.perf_counter() - t0, time.process_time() - cpu_before
                for k in args.k:
                    panel = selected.panels.get(int(k))
                    info = selected.info
                    n_sel = 0 if panel is None else len(panel)
                    fold_rows.append({"cohort": name, "repeat": run.repeat, "seed": run.seed, "fold": run.fold,
                                 "method": method, "k": int(k), "n_train": len(tr), "p0": len(pre),
                                 "n_selected": n_sel, "exact_k": info.get("exact", {}).get(int(k), True) if panel is not None else False,
                                 "n_filled": info.get("n_filled", {}).get(int(k), 0),
                                 "selector_param": info.get("C", {}).get(int(k)),
                                 "n_eval": info.get("n_eval", {}).get(int(k)),
                                 "best_fitness": info.get("best_fitness", {}).get(int(k)),
                                 "selection_time_s": elapsed, "cpu_time_s": cpu_s,
                                 "rss_delta_mb": (process.memory_info().rss - rss_before) / 2**20,
                                 "emissions_kg_co2": emissions_kg,
                                 "missing_genes": "" if panel is not None else "ktsp: k must be even"})
                    if panel is not None:
                        fold_genes.extend({"cohort": name, "repeat": run.repeat, "fold": run.fold, "method": method,
                                      "k": int(k), "rank": rank, "gene": ds.genes[pre[col]]}
                                     for rank, col in enumerate(panel, 1))
                write_csv_atomic(pd.DataFrame(fold_rows), metrics_path)
                write_csv_atomic(pd.DataFrame(fold_genes), genes_path, columns=GENE_COLUMNS)
                print(f"[{name}] repeat {run.repeat} fold {run.fold} {method}: checkpoint saved", flush=True)
        checkpoint_dir = out / "checkpoints"
        run_df = pd.concat([read_csv_safe(p) for p in checkpoint_dir.glob("*.metrics.csv")], ignore_index=True)
        gene_frames = [read_csv_safe(p) for p in checkpoint_dir.glob("*.genes.csv")]
        gene_frames = [g for g in gene_frames if len(g)]
        gene_df = pd.concat(gene_frames, ignore_index=True) if gene_frames else pd.DataFrame(columns=GENE_COLUMNS)
        run_df.to_csv(out / "selection_per_fold.csv", index=False)
        gene_df.to_csv(out / "selected_genes.csv", index=False)
        stability = []
        for key, group in gene_df.groupby(["cohort", "method", "k"], sort=False):
            panels = [set(x.gene) for _, x in group.groupby(["repeat", "fold"])]
            pairs = [jaccard(a, b) for i, a in enumerate(panels) for b in panels[i + 1:]]
            stability.append({"cohort": key[0], "method": key[1], "k": key[2], "n_panels": len(panels),
                              "mean_pairwise_jaccard": float(np.mean(pairs)) if pairs else float("nan")})
        pd.DataFrame(stability).to_csv(out / "selection_stability.csv", index=False)
        gene_df.groupby(["cohort", "method", "k", "gene"], as_index=False).size().rename(
            columns={"size": "selected_in_outer_folds"}).to_csv(out / "selection_gene_frequency.csv", index=False)
        cfg = {"experiment": "E1_feature_selection", "dataset": name, "role": C.DATASETS.get(name), "protocol":
               "precomputed patient-grouped repeated 5x5 outer train/test; selection and label-free variance prefilter use outer train only",
               "split_file": str(path), "split_sha256": sha256(path), "methods": args.methods, "methods_skipped": sorted(skipped),
               "fixed_panels": {m: FIXED_PANELS[m] for m in args.methods if m in FIXED_PANELS}, "k_grid": args.k,
               "prefilter_p0": args.prefilter, "prefilter_rule": args.prefilter_rule, "gene_space": ds.notes, "n_samples": ds.n, "n_genes": ds.p,
               "metrics": ["n_selected", "exact_k", "n_filled", "selection_time_s (descriptive, not E2)", "cpu_time_s", "rss_delta_mb (descriptive)",
                           "emissions_kg_co2 (optional estimate)", "n_eval", "best_fitness", "pairwise_jaccard (descriptive; E4 is the CI version)"],
               "methods_in_results": sorted(run_df.method.unique()) if len(run_df) else [],
               "redo_methods": list(args.redo_methods or []),
               "checkpoints_reused_from_commit": (prev_cfg or {}).get("commit_hash") if args.redo_methods else None,
               "versions": all_versions()}
        write_config(out, prov, cfg, data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv", path])


if __name__ == "__main__":
    main()
