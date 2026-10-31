"""E1 phase B (survey v3 = RQ1): evaluate fixed Phase-A gene panels, B0 and random-k on precomputed 5x5 outer tests.

Approved classifiers only (QĐ13, KBTN 5.2): LR-L2 (main; C by inner 3-fold CV on outer train, threshold on inner OOF)
and, with --secondary, RF and linear SVM at k in SECONDARY_K. The eleven-model zoo of survey_v2/e1_classification.py
runs only with --models-extended, writes to a SEPARATE file (classification_extended_per_fold.csv) and must not be
used for ranking (O19). There is NO "best classifier by test AUROC" output any more: that would select on the test set.

Reference points computed here on the same outer splits (same classifier, same protocol):
* b0       : LR-L2 on all genes of the gene space (no prefilter), one row per fold (k = p);
* random_k : N_RANDOM_DRAWS random panels of size k inside the label-free variance prefilter of the outer train
             (seed [cohort, seed, fold, k, draw], Y14); per-draw rows in e1_random_k.csv, per-fold mean in the main table.

Outputs (results/survey_v2/E1_classification/<cohort>/): classification_per_fold.csv, classification_summary.csv,
e1_random_k.csv, e1_scores.csv (--save-scores), config.json. Re-running resumes (existing rows are skipped).
v3.1.6 (QĐ25): --redo-methods M ... drops the existing rows of those selectors from classification_per_fold.csv (and
e1_scores.csv) before evaluating, so their panels (re-selected in phase A) are evaluated again; everything else is kept.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.classify import evaluate_panel  # noqa: E402
from survey_v2.data import drop_genes, load_dataset, sha256, variance_prefilter  # noqa: E402
from survey_v2.e1_splits import validate_repeated_split_manifest  # noqa: E402
from survey_v2.methods import random_panel  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, previous_config, results_dir, start_run, write_config  # noqa: E402
from scripts.run_e1_feature_selection import split_indices, split_path  # noqa: E402

ROW_KEY = ["repeat", "fold", "method", "k", "model"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS))
    ap.add_argument("--methods", nargs="+", default=None, help="only panels from these selectors (default: all found)")
    ap.add_argument("--k", nargs="+", type=int, default=None)
    ap.add_argument("--repeats", nargs="+", type=int, default=list(range(len(C.REPEAT_SEEDS))))
    ap.add_argument("--selection-roots", nargs="+", type=Path, default=None,
                    help="results roots containing <cohort>/selected_genes.csv (default: E1_feature_selection and E1_wrapper_feature_selection)")
    ap.add_argument("--selection-root", type=Path, default=None,
                    help="(alias kept for the Kaggle runners of the code team, commit 511cbf2) one extra selection root")
    ap.add_argument("--splits-root", type=Path, default=None)
    ap.add_argument("--out", default="E1_classification")
    ap.add_argument("--secondary", action="store_true", help="also RF and linear SVM at k in SECONDARY_K")
    ap.add_argument("--no-references", dest="references", action="store_false", help="skip b0 and random-k")
    ap.add_argument("--random-draws", type=int, default=C.N_RANDOM_DRAWS)
    ap.add_argument("--prefilter", type=int, default=C.PREFILTER_P0)
    ap.add_argument("--save-scores", action="store_true", help="write per-sample test scores of lr_l2 (large)")
    ap.add_argument("--exclude-genes", nargs="+", default=None, help="ablation: B0 và random-k cũng loại các gen này (dùng cùng --selection-roots/--out riêng)")
    ap.add_argument("--models-extended", action="store_true",
                    help="ALSO run the 11-model zoo (descriptive, separate file; never for ranking)")
    ap.add_argument("--quick", action="store_true", help="small grids for the extended zoo smoke run")
    ap.add_argument("--redo-methods", nargs="+", default=None,
                    help="v3.1.6: bỏ các dòng đã có của các phương pháp này rồi đánh giá lại (panel pha A đã chọn lại)")
    ap.set_defaults(references=True)
    return ap.parse_args(argv)


def load_panels(selection_roots, name: str) -> pd.DataFrame:
    frames = []
    for root in selection_roots:
        f = Path(root) / name / "selected_genes.csv"
        if f.exists():
            try:
                g = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                continue
            if len(g):
                g["selection_root"] = str(root)
                frames.append(g)
    if not frames:
        return pd.DataFrame(columns=["cohort", "repeat", "fold", "method", "k", "rank", "gene", "selection_root"])
    return pd.concat(frames, ignore_index=True)


def _row(name, repeat, seed, fold, method, k, res, n_features, t_fit, extra=None):
    return {"cohort": name, "repeat": int(repeat), "seed": int(seed), "fold": int(fold), "method": method, "k": int(k),
            "model": res.classifier, "n_features": int(n_features), "auroc": res.auroc, "bacc": res.bacc, "mcc": res.mcc,
            "logloss": res.logloss, "brier": res.brier, "threshold": res.threshold, "C": res.hyperparam,
            "fit_time_s": t_fit, **(extra or {})}


def drop_rows_of_methods(paths, methods) -> int:
    """v3.1.6: xoá các dòng có cột method thuộc `methods` trong các tệp CSV (nếu có); trả về số dòng đã bỏ."""
    n = 0
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if "method" not in df.columns:
            continue
        keep = df[~df.method.isin(list(methods))]
        n += len(df) - len(keep)
        keep.to_csv(path, index=False, encoding="utf-8-sig")
    return n


def main(argv=None, script=__file__):
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    base = DATA_ROOT / "results" / "survey_v2"
    selection_roots = list(args.selection_roots or [base / "E1_feature_selection", base / "E1_wrapper_feature_selection"])
    if args.selection_root is not None and args.selection_root not in selection_roots:
        selection_roots.append(args.selection_root)
    classifiers_for = (lambda k: list(C.CLASSIFIERS_MAIN) + (list(C.CLASSIFIERS_SECONDARY) if args.secondary and k in C.SECONDARY_K else []))
    for name in args.datasets:
        ds, out = load_dataset(DATA_ROOT, name), results_dir(args.out, name)
        if args.exclude_genes:
            ds = drop_genes(ds, args.exclude_genes)
        manifest_path = split_path(ds, args.splits_root)
        manifest = pd.read_csv(manifest_path)
        errors = validate_repeated_split_manifest(manifest, n_folds=C.N_FOLDS)
        if errors:
            raise ValueError(f"{name}: invalid repeated_splits.csv: {'; '.join(errors)}")
        panels = load_panels(selection_roots, name)
        if args.methods:
            panels = panels[panels.method.isin(args.methods)]
        if args.k:
            panels = panels[panels.k.isin(args.k)]
        panels = panels[panels.repeat.isin(args.repeats)]
        gene_index = {g: i for i, g in enumerate(ds.genes)}
        per_fold_path, random_path, scores_path = out / "classification_per_fold.csv", out / "e1_random_k.csv", out / "e1_scores.csv"
        prev_cfg = previous_config(out)
        if args.redo_methods:
            n_drop = drop_rows_of_methods([per_fold_path, scores_path], args.redo_methods)
            print(f"[{name}] --redo-methods {args.redo_methods}: đã bỏ {n_drop} dòng cũ, sẽ đánh giá lại", flush=True)
        done = set()
        if per_fold_path.exists():
            prev = pd.read_csv(per_fold_path)
            done = set(map(tuple, prev[ROW_KEY].itertuples(index=False, name=None)))
        rows, rand_rows, score_rows = [], [], []
        cid = C.cohort_id(name)
        k_grid = args.k or list(C.K_GRID)
        runs = manifest[["repeat", "seed", "fold"]].drop_duplicates().sort_values(["repeat", "fold"])
        for run in runs.itertuples(index=False):
            if run.repeat not in args.repeats:
                continue
            repeat, seed, fold = int(run.repeat), int(run.seed), int(run.fold)
            tr, te = split_indices(ds, manifest, repeat, fold)
            Xtr, ytr, Xte, yte = ds.X[tr], ds.y[tr], ds.X[te], ds.y[te]
            with threadpool_limits(1):
                # ---- tham chiếu: B0 và random-k (cùng lần chia, cùng bộ phân loại) ------------------------------
                if args.references:
                    if (repeat, fold, "b0", ds.p, "lr_l2") not in done:
                        t0 = time.perf_counter()
                        res = evaluate_panel(Xtr, ytr, Xte, yte, classifier="lr_l2", seed=C.seed_for(cid, seed, fold, 0))
                        rows.append(_row(name, repeat, seed, fold, "b0", ds.p, res, ds.p, time.perf_counter() - t0))
                        if args.save_scores:
                            score_rows += [{"cohort": name, "repeat": repeat, "fold": fold, "method": "b0", "k": ds.p, "sample_id": s,
                                            "label": int(l), "score": float(sc)} for s, l, sc in zip(ds.sample_ids[te], yte, res.scores_test)]
                    pre = variance_prefilter(Xtr, args.prefilter)
                    for k in k_grid:
                        if (repeat, fold, "random_k", k, "lr_l2") in done or k > len(pre):
                            continue
                        aucs = []
                        for draw in range(args.random_draws):
                            rng = np.random.default_rng([cid, seed, fold, int(k), draw])
                            cols = pre[random_panel(len(pre), int(k), rng)]
                            res = evaluate_panel(Xtr[:, cols], ytr, Xte[:, cols], yte, classifier="lr_l2",
                                                 seed=C.seed_for(cid, seed, fold, int(k), draw))
                            rand_rows.append({"cohort": name, "repeat": repeat, "seed": seed, "fold": fold, "k": int(k), "draw": draw,
                                              "auroc": res.auroc, "bacc": res.bacc, "mcc": res.mcc, "logloss": res.logloss, "brier": res.brier})
                            aucs.append(res)
                        mean = lambda attr: float(np.mean([getattr(r, attr) for r in aucs]))  # noqa: E731
                        rows.append({"cohort": name, "repeat": repeat, "seed": seed, "fold": fold, "method": "random_k", "k": int(k),
                                     "model": "lr_l2", "n_features": int(k), "auroc": mean("auroc"), "bacc": mean("bacc"),
                                     "mcc": mean("mcc"), "logloss": mean("logloss"), "brier": mean("brier"), "threshold": np.nan,
                                     "C": np.nan, "fit_time_s": np.nan, "n_draws": args.random_draws})
                # ---- panel của pha A ------------------------------------------------------------------------------
                sub = panels[(panels.repeat == repeat) & (panels.fold == fold)]
                for (method, k), group in sub.groupby(["method", "k"], sort=True):
                    genes = group.sort_values("rank").gene.tolist()
                    if len(genes) != len(set(genes)) or any(g not in gene_index for g in genes):
                        raise ValueError(f"{name} repeat={repeat} fold={fold} {method} k={k}: invalid fixed panel")
                    cols = [gene_index[g] for g in genes]
                    for model in classifiers_for(int(k)):
                        if (repeat, fold, method, int(k), model) in done:
                            continue
                        t0 = time.perf_counter()
                        res = evaluate_panel(Xtr[:, cols], ytr, Xte[:, cols], yte, classifier=model,
                                             seed=C.seed_for(cid, seed, fold, int(k), C.METHOD_ID.get(method, 90)))
                        rows.append(_row(name, repeat, seed, fold, method, int(k), res, len(cols), time.perf_counter() - t0,
                                         {"n_features_vs_k": len(cols) - int(k)}))
                        if args.save_scores and model == "lr_l2":
                            score_rows += [{"cohort": name, "repeat": repeat, "fold": fold, "method": method, "k": int(k), "sample_id": s,
                                            "label": int(l), "score": float(sc)} for s, l, sc in zip(ds.sample_ids[te], yte, res.scores_test)]
                        print(f"[{name}] repeat {repeat} fold {fold} {method} k={k} {model}: AUROC={res.auroc:.3f}", flush=True)
            # ghi dần để resume được
            if rows:
                pd.DataFrame(rows).to_csv(per_fold_path, mode="a", index=False, header=not per_fold_path.exists(), encoding="utf-8-sig")
                rows = []
            if rand_rows:
                pd.DataFrame(rand_rows).to_csv(random_path, mode="a", index=False, header=not random_path.exists(), encoding="utf-8-sig")
                rand_rows = []
            if score_rows:
                pd.DataFrame(score_rows).to_csv(scores_path, mode="a", index=False, header=not scores_path.exists())
                score_rows = []
        if not per_fold_path.exists():
            print(f"[{name}] không có gì để đánh giá (chưa có panel pha A?)", flush=True)
            continue
        result_df = pd.read_csv(per_fold_path).drop_duplicates(subset=ROW_KEY, keep="last")
        result_df.to_csv(per_fold_path, index=False, encoding="utf-8-sig")
        metric_cols = ["auroc", "bacc", "mcc", "logloss", "brier", "fit_time_s"]
        summary = result_df.groupby(["cohort", "method", "k", "model"], as_index=False)[metric_cols].agg(["mean", "std", "count"])
        summary.columns = ["_".join(c for c in col if c) for col in summary.columns.to_flat_index()]
        summary.to_csv(out / "classification_summary.csv", index=False, encoding="utf-8-sig")
        cfg = {"experiment": "E1_classification", "dataset": name, "role": C.DATASETS.get(name), "protocol":
               "fixed Phase-A panels; precomputed patient-grouped repeated 5x5 outer tests; C and threshold by inner 3-fold CV on outer train only; "
               "b0 = LR-L2 on all genes; random_k = mean over draws inside the label-free prefilter of the outer train",
               "selection_roots": [str(r) for r in selection_roots], "split_file": str(manifest_path), "split_sha256": sha256(manifest_path),
               "classifiers_main": list(C.CLASSIFIERS_MAIN), "classifiers_secondary": list(C.CLASSIFIERS_SECONDARY) if args.secondary else [],
               "secondary_k": list(C.SECONDARY_K), "random_k_draws": args.random_draws, "prefilter_p0": args.prefilter,
               "small_n_rule": f"C = {C.SMALL_N_FIXED_C} fixed and threshold 0.5 when the smaller class has <= {C.SMALL_N_T_MAX} training samples (M5)",
               "metrics": ["auroc", "bacc", "mcc", "logloss", "brier"], "inner_cv_folds": C.INNER_CV_FOLDS,
               "gene_space": ds.notes, "n_samples": ds.n, "n_genes": ds.p, "methods_found": sorted(panels.method.unique()),
               "note": "no best-classifier-by-test file (O19); extended zoo only in classification_extended_per_fold.csv",
               "methods_in_results": sorted(result_df.method.unique()), "redo_methods": list(args.redo_methods or []),
               "rows_reused_from_commit": (prev_cfg or {}).get("commit_hash") if args.redo_methods else None,
               "versions": all_versions()}
        write_config(out, prov, cfg, data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv", manifest_path])
        # ---- (tuỳ chọn) bộ 11 mô hình — mô tả, không dùng xếp hạng -----------------------------------------------
        if args.models_extended:
            from survey_v2.e1_classification import CLASSIFICATION_MODELS, tune_and_evaluate
            print(f"[{name}] CẢNH BÁO: bộ mô hình mở rộng chỉ để mô tả (phụ lục), không dùng để xếp hạng (QĐ13, O19)", flush=True)
            ext_rows = []
            for run in runs.itertuples(index=False):
                if run.repeat not in args.repeats:
                    continue
                repeat, seed, fold = int(run.repeat), int(run.seed), int(run.fold)
                tr, te = split_indices(ds, manifest, repeat, fold)
                sub = panels[(panels.repeat == repeat) & (panels.fold == fold)]
                for (method, k), group in sub.groupby(["method", "k"], sort=True):
                    cols = [gene_index[g] for g in group.sort_values("rank").gene.tolist()]
                    for model in CLASSIFICATION_MODELS:
                        with threadpool_limits(1):
                            r = tune_and_evaluate(model, ds.X[tr][:, cols], ds.y[tr], ds.X[te][:, cols], ds.y[te],
                                                  seed=C.seed_for(cid, seed, fold, len(cols)), quick=args.quick)
                        ext_rows.append({"cohort": name, "repeat": repeat, "seed": seed, "fold": fold, "method": method, "k": int(k),
                                         "n_features": len(cols), **r})
            pd.DataFrame(ext_rows).to_csv(out / "classification_extended_per_fold.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
