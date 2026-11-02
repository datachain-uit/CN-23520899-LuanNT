"""E7 — ba chiến lược chuyển giao ĐƠN GIẢN nguồn TCGA → cohort GEO trên đúng tập con few-shot của E1b (survey v3, ưu tiên 2).

Chiến lược (survey_v2/transfer.py; = baseline D3 của RQ2 bài phương pháp): source_only (zero-shot, chuẩn hoá mức panel φ,
không dùng nhãn đích), select_on_source (panel từ nguồn, fit LR-L2 trên L_T), pooled (nguồn ⊕ L_T, z-score theo từng bộ,
chọn gen trên ma trận gộp, fit trên L_T) và target_only (ghép cặp). Bộ chọn gen chung: TRANSFER_SELECTOR (L1-LR đúng k).
Lưới: n_T ∈ TRANSFER_N_T = {5, 10, 20}; k ∈ TRANSFER_K_GRID = {2, 3, 4, 5, 10}; 3 cohort GPL570 so sánh, GSE42127 mô tả.
Không gian gen G = giao gen cohort ∩ TCGA; lọc sơ bộ không nhãn tính trên NGUỒN cho ba chiến lược chuyển giao, trên L_T cho
target_only. Không có MVP, ptLasso, Trans-GLM ở đây (bài 2).

    python scripts/run_survey_e7_transfer_naive.py --n-jobs 4
    python scripts/run_survey_e7_transfer_naive.py --datasets gse37745 --n-t 5 --k 2 5

Đầu ra: results/survey_v2/E7/<cohort>/{e7_per_run.csv, e7_selected_genes.csv, config.json}. Resume theo (split, subset, n_t, strategy, k).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import load_dataset, sha256  # noqa: E402
from survey_v2.fewshot import fewshot_runs, validate_fewshot_manifest  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402
from survey_v2.transfer import TransferContext  # noqa: E402
from scripts.build_fewshot_splits import fewshot_path  # noqa: E402
from scripts.run_e1_classification import append_csv_aligned  # noqa: E402

RUN_KEY = ["split", "subset", "n_t", "strategy", "k"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.FEWSHOT_COHORTS))
    ap.add_argument("--source", default=C.SOURCE_DATASET)
    ap.add_argument("--strategies", nargs="+", default=list(C.TRANSFER_STRATEGIES))
    ap.add_argument("--k", nargs="+", type=int, default=list(C.TRANSFER_K_GRID))
    ap.add_argument("--n-t", nargs="+", default=[str(v) for v in C.TRANSFER_N_T])
    ap.add_argument("--selector", default=C.TRANSFER_SELECTOR)
    ap.add_argument("--prefilter", type=int, default=C.PREFILTER_P0)
    ap.add_argument("--splits-root", type=Path, default=None)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", default="E7")
    return ap.parse_args(argv)


def _res_row(base, strategy, k, res, genes, extra=None):
    return {**base, "strategy": strategy, "k": int(k), "auroc": res.auroc, "bacc": res.bacc, "mcc": res.mcc, "logloss": res.logloss,
            "brier": res.brier, "C": res.hyperparam, "n_selected": len(genes), **(extra or {})}


def run_one(ctx: TransferContext, ds, cid, split, seed, subset, n_t, train_ids, test_ids, strategies, k_list, done):
    index = {s: i for i, s in enumerate(ds.sample_ids)}
    tr = np.array([index[s] for s in train_ids]); te = np.array([index[s] for s in test_ids])
    base = {"cohort": ds.name, "split": split, "seed": seed, "subset": subset, "n_t": n_t, "n_train": len(tr), "n_test": len(te)}
    n_t_int = 10**6 if n_t == "pool" else int(n_t)
    rows, gene_rows = [], []
    with threadpool_limits(1):
        for k in k_list:
            seed_k = C.seed_for(cid, seed, subset, n_t_int, k, 77)
            for strategy in strategies:
                if (split, subset, n_t, strategy, k) in done:
                    continue
                t0 = time.perf_counter()
                if strategy == "source_only":
                    res = ctx.source_only(k, te, seed_k)
                    genes = ctx.genes[ctx.source_panels[k]]
                    extra = {"depends_on_n_t": False, "panel_normalisation": C.PANEL_NORMALISATION}
                elif strategy == "select_on_source":
                    res = ctx.select_on_source(k, tr, te, seed_k)
                    genes = ctx.genes[ctx.source_panels[k]]
                    extra = {"depends_on_n_t": True}
                elif strategy == "pooled":
                    res, S = ctx.pooled(k, tr, te, seed_k)
                    genes = ctx.genes[S]
                    extra = {"depends_on_n_t": True}
                elif strategy == "target_only":
                    res, S, info = ctx.target_only(k, tr, te, seed_k)
                    genes = ctx.genes[S]
                    extra = {"depends_on_n_t": True, "n_filled": info.get("n_filled", {}).get(k, 0), "exact_k": info.get("exact", {}).get(k, True)}
                else:
                    raise ValueError(strategy)
                rows.append(_res_row(base, strategy, k, res, genes, {**extra, "fit_time_s": time.perf_counter() - t0}))
                gene_rows += [{**base, "strategy": strategy, "k": int(k), "rank": r + 1, "gene": g} for r, g in enumerate(genes)]
    return rows, gene_rows


def append_csv(path: Path, rows):
    if rows:
        append_csv_aligned(pd.DataFrame(rows), path)        # v3.1.7 (Y31): nối theo đúng thứ tự cột của tiêu đề đã có


def main(argv=None, script=__file__):
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t_all = time.time()
    source = load_dataset(DATA_ROOT, args.source, gene_space=None)
    for name in args.datasets:
        ds, out = load_dataset(DATA_ROOT, name), results_dir(args.out, name)
        cid = C.cohort_id(name)
        path = fewshot_path(ds, args.splits_root)
        man = pd.read_csv(path, dtype={"n_t": str})
        errors = validate_fewshot_manifest(man)
        if errors:
            raise ValueError(f"{name}: fewshot_splits.csv không hợp lệ: {errors}")
        with threadpool_limits(1):
            ctx = TransferContext(source, ds, args.k, selector=args.selector, prefilter=args.prefilter, seed=C.seed_for(cid, 0, 0, 7))
        per_run = out / "e7_per_run.csv"
        done = set()
        if per_run.exists():
            prev = pd.read_csv(per_run, dtype={"n_t": str})
            done = set(map(tuple, prev[RUN_KEY].itertuples(index=False, name=None)))
        jobs = [(s, se, su, nt, tr, te) for s, se, su, nt, tr, te in fewshot_runs(man) if nt in args.n_t]
        print(f"[{name}] E7: {len(jobs)} lần chạy × {len(args.k)} k × {len(args.strategies)} chiến lược (trừ phần đã có); |G| = {len(ctx.genes)}", flush=True)
        for rows, gene_rows in Parallel(n_jobs=args.n_jobs, backend="loky", return_as="generator")(
                delayed(run_one)(ctx, ds, cid, s, se, su, nt, tr, te, args.strategies, args.k, done) for s, se, su, nt, tr, te in jobs):
            append_csv(per_run, rows)
            append_csv(out / "e7_selected_genes.csv", gene_rows)
        if not per_run.exists():
            continue
        df = pd.read_csv(per_run, dtype={"n_t": str})
        cfg = {"experiment": "E7_transfer_naive", "dataset": name, "role": C.DATASETS.get(name), "source": args.source,
               "source_sha256": source.files_sha256, "n_source": source.n, "gene_space_size": int(len(ctx.genes)),
               "strategies": args.strategies, "selector": args.selector, "k_grid": args.k, "n_t_levels": args.n_t,
               "prefilter_p0_on_source": int(len(ctx.pre)), "panel_normalisation": C.PANEL_NORMALISATION,
               "source_panels": {int(k): ctx.genes[v].tolist() for k, v in ctx.source_panels.items()},
               "source_selector_info": {kk: (vv if not hasattr(vv, "tolist") else vv.tolist()) for kk, vv in
                                        {k: ctx.source_info.get(k) for k in ("C", "exact", "n_filled")}.items()},
               "fewshot_splits_file": str(path), "fewshot_splits_sha256": sha256(path), "n_rows": int(len(df)),
               "note": "no proposed method here; ptLasso / Trans-GLM / MVP belong to the method paper",
               "elapsed_s": time.time() - t_all, "versions": all_versions()}
        write_config(out, prov, cfg, data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv", path,
                                                 source.data_dir / "expression.h5", source.data_dir / "metadata.csv"])
        print(f"[{name}] xong: {len(df)} dòng", flush=True)


if __name__ == "__main__":
    main()
