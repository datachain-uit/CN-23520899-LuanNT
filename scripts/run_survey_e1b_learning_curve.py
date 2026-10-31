"""E1b — đường cong học few-shot của các phương pháp chọn gen HIỆN CÓ (survey v3; = nhánh target-only của RQ2).

Giao thức (KBTN v2 Mục 6; bản chia từ scripts/build_fewshot_splits.py): mỗi cohort GEO, 5 lần chia test_T / pool_T,
4 tập con mỗi mức n_T ∈ {5, 10, 20, 40, pool} mỗi lớp (lồng nhau), k ∈ FEWSHOT_K_GRID = {2, 3, 4, 5, 10}.
Trên mỗi tập con L_T: lọc phương sai (không nhãn) trên L_T → 11 phương pháp không wrapper (FEWSHOT_METHODS) → LR-L2
(M5: n_T ≤ 10 thì C = 1 cố định, ngưỡng 0,5; n_T ≥ 20 thì CV nội bộ) → đánh giá trên test_T. Thêm random-k
(N_RANDOM_DRAWS lần/tập con) và panel IHC-2/IHC-4 (cố định) để đối chiếu. Mọi phương pháp dùng chung tập con (ghép cặp).
Độ ổn định few-shot: Nogueira (+ CI tham khảo) qua 20 lần chạy (5 split × 4 subset) ở mỗi (n_T, method, k), không gian = G.

    python scripts/run_survey_e1b_learning_curve.py --n-jobs 4
    python scripts/run_survey_e1b_learning_curve.py --datasets gse37745 --methods f_test l1_lr --n-t 5 10 --k 2 5

Đầu ra: results/survey_v2/E1b/<cohort>/{e1b_per_run.csv, e1b_selected_genes.csv, e1b_random_k.csv, e1b_stability.csv, config.json}.
Resume: (split, subset, n_t, method) đã có trong e1b_per_run.csv được bỏ qua.
v3.1.6 (QĐ25): --redo-methods M ... bỏ các dòng của các phương pháp này trong e1b_per_run.csv và e1b_selected_genes.csv rồi
chạy lại chúng (phần khác giữ nguyên); e1b_stability.csv được tính lại từ toàn bộ gen đã chọn.
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
from survey_v2.classify import evaluate_panel  # noqa: E402
from survey_v2.data import load_dataset, sha256, variance_prefilter  # noqa: E402
from survey_v2.fewshot import fewshot_runs, n_t_sort_key, validate_fewshot_manifest  # noqa: E402
from survey_v2.methods import random_panel, select  # noqa: E402
from survey_v2.panels import all_fixed_panels, panel_indices  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, previous_config, results_dir, start_run, write_config  # noqa: E402
from survey_v2.stability import nogueira, selections_to_matrix  # noqa: E402
from scripts.build_fewshot_splits import fewshot_path  # noqa: E402
from scripts.run_e1_classification import drop_rows_of_methods  # noqa: E402

FIXED_PANELS = all_fixed_panels()
RUN_KEY = ["split", "subset", "n_t", "method"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.FEWSHOT_COHORTS))
    ap.add_argument("--methods", nargs="+", default=list(C.FEWSHOT_METHODS) + ["ihc_2", "ihc_4"])
    ap.add_argument("--k", nargs="+", type=int, default=list(C.FEWSHOT_K_GRID))
    ap.add_argument("--n-t", nargs="+", default=None, help="mức n_T cần chạy (số hoặc 'pool'); mặc định mọi mức trong bản chia")
    ap.add_argument("--splits-root", type=Path, default=None)
    ap.add_argument("--prefilter", type=int, default=C.PREFILTER_P0)
    ap.add_argument("--random-draws", type=int, default=C.N_RANDOM_DRAWS)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", default="E1b")
    ap.add_argument("--redo-methods", nargs="+", default=None, help="v3.1.6: bỏ dòng cũ của các phương pháp này rồi chạy lại")
    return ap.parse_args(argv)


def run_one(ds, cid, split, seed, subset, n_t, train_ids, test_ids, methods, k_list, args):
    index = {s: i for i, s in enumerate(ds.sample_ids)}
    tr = np.array([index[s] for s in train_ids]); te = np.array([index[s] for s in test_ids])
    Xtr, ytr, Xte, yte = ds.X[tr], ds.y[tr], ds.X[te], ds.y[te]
    base = {"cohort": ds.name, "split": split, "seed": seed, "subset": subset, "n_t": n_t, "n_train": len(tr), "n_test": len(te)}
    rows, gene_rows, rand_rows = [], [], []
    n_t_int = 10**6 if n_t == "pool" else int(n_t)
    with threadpool_limits(1):
        pre = variance_prefilter(Xtr, args.prefilter)
        ks = [k for k in k_list if k <= len(pre)]
        for method in methods:
            t0 = time.perf_counter()
            if method in FIXED_PANELS:
                cols, missing = panel_indices(ds.genes, FIXED_PANELS[method])
                k_panel = len(FIXED_PANELS[method])
                if len(cols) == 0:
                    continue
                res = evaluate_panel(Xtr[:, cols], ytr, Xte[:, cols], yte, classifier="lr_l2", seed=C.seed_for(cid, seed, subset, n_t_int, 90))
                rows.append({**base, "method": method, "k": k_panel, "n_selected": len(cols), "exact_k": not missing, "n_filled": 0,
                             "auroc": res.auroc, "bacc": res.bacc, "mcc": res.mcc, "logloss": res.logloss, "brier": res.brier,
                             "C": res.hyperparam, "select_time_s": 0.0, "missing_genes": ";".join(missing)})
                gene_rows += [{**base, "method": method, "k": k_panel, "rank": r + 1, "gene": ds.genes[c]} for r, c in enumerate(cols)]
                continue
            if method == "random_k":
                for k in ks:
                    aucs = []
                    for draw in range(args.random_draws):
                        rng = np.random.default_rng([cid, seed, subset, n_t_int, k, draw])
                        cols = pre[random_panel(len(pre), k, rng)]
                        r = evaluate_panel(Xtr[:, cols], ytr, Xte[:, cols], yte, classifier="lr_l2", seed=C.seed_for(cid, seed, subset, n_t_int, k, draw))
                        rand_rows.append({**base, "k": k, "draw": draw, "auroc": r.auroc, "bacc": r.bacc, "mcc": r.mcc, "logloss": r.logloss, "brier": r.brier})
                        aucs.append(r)
                    rows.append({**base, "method": "random_k", "k": k, "n_selected": k, "exact_k": True, "n_filled": 0,
                                 "auroc": float(np.mean([r.auroc for r in aucs])), "bacc": float(np.mean([r.bacc for r in aucs])),
                                 "mcc": float(np.mean([r.mcc for r in aucs])), "logloss": float(np.mean([r.logloss for r in aucs])),
                                 "brier": float(np.mean([r.brier for r in aucs])), "C": np.nan, "select_time_s": 0.0, "missing_genes": ""})
                continue
            try:
                sel = select(method, Xtr[:, pre], ytr, ks, seed=C.seed_for(cid, seed, subset, n_t_int, C.METHOD_ID.get(method, 50)))
            except ImportError as exc:
                print(f"[{ds.name}] {method}: BỎ QUA ({exc})", flush=True)
                continue
            dt = time.perf_counter() - t0
            for k in ks:
                panel = sel.panels.get(k)
                if panel is None:                      # k-TSP ở k lẻ
                    rows.append({**base, "method": method, "k": k, "n_selected": 0, "exact_k": False, "n_filled": 0, "auroc": np.nan,
                                 "bacc": np.nan, "mcc": np.nan, "logloss": np.nan, "brier": np.nan, "C": np.nan, "select_time_s": dt,
                                 "missing_genes": "ktsp: k must be even"})
                    continue
                cols = pre[np.asarray(panel)]
                res = evaluate_panel(Xtr[:, cols], ytr, Xte[:, cols], yte, classifier="lr_l2", seed=C.seed_for(cid, seed, subset, n_t_int, k))
                rows.append({**base, "method": method, "k": k, "n_selected": len(cols), "exact_k": sel.info.get("exact", {}).get(k, True),
                             "n_filled": sel.info.get("n_filled", {}).get(k, 0), "auroc": res.auroc, "bacc": res.bacc, "mcc": res.mcc,
                             "logloss": res.logloss, "brier": res.brier, "C": res.hyperparam, "select_time_s": dt, "missing_genes": ""})
                gene_rows += [{**base, "method": method, "k": k, "rank": r + 1, "gene": ds.genes[c]} for r, c in enumerate(cols)]
    return rows, gene_rows, rand_rows


def append_csv(path: Path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=not path.exists(), encoding="utf-8-sig")


def fewshot_stability(genes: pd.DataFrame, universe) -> pd.DataFrame:
    rows = []
    for (n_t, method, k), g in genes.groupby(["n_t", "method", "k"], sort=False):
        sets = [list(dict.fromkeys(x.gene)) for _, x in g.groupby(["split", "subset"])]
        if len(sets) < 2:
            continue
        nog = nogueira(selections_to_matrix(sets, list(universe)))
        rows.append({"cohort": genes.cohort.iloc[0], "n_t": n_t, "method": method, "k": int(k), "n_runs": len(sets), "d": len(universe),
                     "nogueira": nog["stability"], "nogueira_ci_low": nog["ci_low"], "nogueira_ci_high": nog["ci_high"],
                     "note": "CI tham khảo: các tập con cùng split chồng lấn (lồng nhau theo n_T)"})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["method", "k", "n_t"], key=lambda s: s.map(n_t_sort_key) if s.name == "n_t" else s)
    return df


def main(argv=None, script=__file__):
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t_all = time.time()
    methods = list(args.methods) + ["random_k"]
    for name in args.datasets:
        ds, out = load_dataset(DATA_ROOT, name), results_dir(args.out, name)
        cid = C.cohort_id(name)
        path = fewshot_path(ds, args.splits_root)
        man = pd.read_csv(path, dtype={"n_t": str})
        errors = validate_fewshot_manifest(man)
        if errors:
            raise ValueError(f"{name}: fewshot_splits.csv không hợp lệ: {errors}")
        per_run = out / "e1b_per_run.csv"
        prev_cfg = previous_config(out)
        if args.redo_methods:
            n_drop = drop_rows_of_methods([per_run, out / "e1b_selected_genes.csv"], args.redo_methods)
            print(f"[{name}] --redo-methods {args.redo_methods}: đã bỏ {n_drop} dòng cũ, sẽ chạy lại", flush=True)
        done = set()
        if per_run.exists():
            prev = pd.read_csv(per_run, dtype={"n_t": str})
            done = set(map(tuple, prev[RUN_KEY].itertuples(index=False, name=None)))
        jobs = []
        for split, seed, subset, n_t, train_ids, test_ids in fewshot_runs(man):
            if args.n_t and n_t not in args.n_t:
                continue
            todo = [m for m in methods if (split, subset, n_t, m) not in done]
            if todo:
                jobs.append((split, seed, subset, n_t, train_ids, test_ids, todo))
        print(f"[{name}] E1b: {len(jobs)} lần chạy (split, subset, n_t) cần làm", flush=True)
        for rows, gene_rows, rand_rows in Parallel(n_jobs=args.n_jobs, backend="loky", return_as="generator")(
                delayed(run_one)(ds, cid, s, se, su, nt, tr, te, todo, args.k, args) for s, se, su, nt, tr, te, todo in jobs):
            append_csv(per_run, rows)
            append_csv(out / "e1b_selected_genes.csv", gene_rows)
            append_csv(out / "e1b_random_k.csv", rand_rows)
        if not per_run.exists():
            continue
        runs_df = pd.read_csv(per_run, dtype={"n_t": str})
        stab = pd.DataFrame()
        if (out / "e1b_selected_genes.csv").exists():
            genes = pd.read_csv(out / "e1b_selected_genes.csv", dtype={"n_t": str})
            stab = fewshot_stability(genes, ds.genes)
            stab.to_csv(out / "e1b_stability.csv", index=False, encoding="utf-8-sig")
        cfg = {"experiment": "E1b_learning_curve", "dataset": name, "role": C.DATASETS.get(name), "methods": methods, "k_grid": args.k,
               "n_t_levels": sorted(runs_df.n_t.unique(), key=n_t_sort_key), "n_runs": int(runs_df.groupby(RUN_KEY[:3]).ngroups),
               "protocol": "5 test_T/pool_T splits x 4 nested subsets per n_T per class; prefilter and selection on L_T only; LR-L2 with M5 small-n rule",
               "fewshot_splits_file": str(path), "fewshot_splits_sha256": sha256(path), "prefilter_p0": args.prefilter,
               "random_k_draws": args.random_draws, "stability": "Nogueira over the runs per (n_T, method, k); universe = gene space G",
               "methods_in_results": sorted(runs_df.method.unique()), "redo_methods": list(args.redo_methods or []),
               "rows_reused_from_commit": (prev_cfg or {}).get("commit_hash") if args.redo_methods else None,
               "gene_space": ds.notes, "n_samples": ds.n, "n_genes": ds.p, "elapsed_s": time.time() - t_all, "versions": all_versions()}
        write_config(out, prov, cfg, data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv", path])
        print(f"[{name}] xong: {len(runs_df)} dòng; ổn định {len(stab)} dòng", flush=True)


if __name__ == "__main__":
    main()
