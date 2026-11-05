"""RQ2: MVP trên đúng các tập con few-shot của E1b và E7 (Review 03/11 Mục 7.3). Bước 0–6: mọi γ cố định và γ chọn theo M5.

Chưa có: thử nghiệm sớm --early-tcga (bước 8).

Mỗi (split, subset, n_t, k) ghi một dòng gamma_rule = "fixed" cho từng γ (để so γ cố định với CV mà không chạy lại) và, với
--gamma-rule m5_cv (mặc định), một dòng gamma_rule = "m5_cv" mang chỉ số của đúng γ đã chọn. Khi tổng hợp phải lọc theo gamma_rule.

Lọc sơ bộ: γ > 0 lọc trên nguồn (ctx.pre, nơi có π_j); γ = 0 là target-only nên lọc trên L_T như E1b, để γ = 0 trùng E1b.

    python scripts/run_rq2_mvp.py --datasets gse37745 --k 2 3 4 5 10 --gamma 0 0.5 1 2 --n-t 5 10 --save-scores

Đầu ra: results/survey_v2/RQ2_mvp/<cohort>/{rq2_mvp_per_run.csv, rq2_mvp_gamma_path.csv, rq2_mvp_selected_genes.csv, rq2_source_pi.csv, config.json}
và rq2_mvp_scores.csv khi có --save-scores. Resume theo (split, subset, n_t, gamma, k) với dòng fixed, (split, subset, n_t, k) với dòng m5_cv.
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
from survey_v2.fewshot import fewshot_runs, validate_fewshot_manifest  # noqa: E402
from survey_v2.mvp import GAMMA_MARGIN, PI_EPSILON, gamma_path, pick_gamma, select_weighted, source_stability  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402
from survey_v2.transfer import TransferContext  # noqa: E402
from scripts.build_fewshot_splits import fewshot_path  # noqa: E402
from scripts.run_e1_classification import append_csv_aligned  # noqa: E402

RUN_KEY = ["split", "subset", "n_t", "gamma", "k"]
GAMMA_GRID = (0.0, 0.5, 1.0, 2.0)
GAMMA_RULES = ("m5_cv", "fixed")
GAMMA_SEED_TAG = 9             # seed của CV chọn γ: seed_for(cid, seed, subset, n_t, 9)
PREFILTER_RULE = "gamma > 0: top variance on source; gamma = 0: top variance on L_T, identical to E1b target-only"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.FEWSHOT_COHORTS))
    ap.add_argument("--source", default=C.SOURCE_DATASET)
    ap.add_argument("--k", nargs="+", type=int, default=list(C.FEWSHOT_K_GRID))
    ap.add_argument("--gamma", nargs="+", type=float, default=list(GAMMA_GRID))
    ap.add_argument("--gamma-rule", choices=GAMMA_RULES, default=GAMMA_RULES[0], help="m5_cv: chọn γ bằng CV trong L_T; fixed: chỉ chạy từng γ")
    ap.add_argument("--n-t", nargs="+", default=None, help="mức n_T cần chạy (số hoặc 'pool'); mặc định mọi mức trong bản chia")
    ap.add_argument("--prefilter", type=int, default=C.PREFILTER_P0)
    ap.add_argument("--splits-root", type=Path, default=None)
    ap.add_argument("--save-scores", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", default="RQ2_mvp")
    return ap.parse_args(argv)


def run_one(ctx: TransferContext, ds, cid, pi, split, seed, subset, n_t, train_ids, test_ids, k_list, gammas, done, prefilter, save_scores):
    """Một (split, subset, n_t). `pi` theo thứ tự cột của ctx.pre. Không có bước nào đọc nhãn test ngoài evaluate_panel."""
    index = {s: i for i, s in enumerate(ds.sample_ids)}
    tr = np.array([index[s] for s in train_ids]); te = np.array([index[s] for s in test_ids])
    Xtr, ytr, Xte, yte = ctx.Xt[tr], ctx.yt[tr], ctx.Xt[te], ctx.yt[te]
    n_t_int = 10**6 if n_t == "pool" else int(n_t)
    base = {"cohort": ds.name, "split": split, "seed": seed, "subset": subset, "n_t": n_t}
    pi_by_gene = np.full(len(ctx.genes), np.nan)
    pi_by_gene[ctx.pre] = pi
    ihc = set(C.IHC_4)
    seed_sel = C.seed_for(cid, seed, subset, n_t_int, C.METHOD_ID["l1_lr"])     # cùng seed với E1b; ghép cặp giữa các γ
    rows, gene_rows, score_rows = [], [], []
    with threadpool_limits(1):
        for gamma in gammas:
            todo = [k for k in k_list if (split, subset, n_t, gamma, k) not in done]
            if not todo:
                continue
            t0 = time.perf_counter()
            if gamma == 0:
                pre, pi_pre = variance_prefilter(Xtr, prefilter), None
            else:
                pre, pi_pre = ctx.pre, pi
            ks = [k for k in todo if k <= len(pre)]
            sel = select_weighted(Xtr[:, pre], ytr, np.zeros(len(pre)) if pi_pre is None else pi_pre, gamma, ks, seed_sel)
            dt = time.perf_counter() - t0
            for k in ks:
                cols = pre[np.asarray(sel.panels[k])]
                res = evaluate_panel(Xtr[:, cols], ytr, Xte[:, cols], yte, classifier="lr_l2", seed=C.seed_for(cid, seed, subset, n_t_int, k))
                genes = ctx.genes[cols]
                rows.append({**base, "k": k, "method": "mvp", "gamma": gamma, "gamma_rule": "fixed", "auroc": res.auroc, "bacc": res.bacc,
                             "mcc": res.mcc, "logloss": res.logloss, "brier": res.brier, "C": res.hyperparam, "threshold": res.threshold,
                             "n_selected": len(cols), "exact_k": sel.info["exact"][k], "n_filled": sel.info["n_filled"][k],
                             "n_ihc_in_panel": sum(g in ihc for g in genes), "select_time_s": dt})
                gene_rows += [{**base, "k": k, "gamma": gamma, "rank": r + 1, "gene": g, "pi_source": pi_by_gene[c]}
                              for r, (g, c) in enumerate(zip(genes, cols))]
                if save_scores:
                    score_rows += [{**base, "k": k, "gamma": gamma, "sample_id": s, "y_true": int(y), "score": float(v)}
                                   for s, y, v in zip(test_ids, yte, res.scores_test)]
    return rows, gene_rows, score_rows


def run_job(ctx: TransferContext, ds, cid, pi, split, seed, subset, n_t, train_ids, test_ids, k_list, gammas, done_fixed, done_cv, prefilter, save_scores, gamma_rule):
    """Một (split, subset, n_t) theo gamma_rule. Đường γ chỉ nhận L_T, nên không thể đọc test_T. Trả về (rows, gene_rows, score_rows, path_rows)."""
    if gamma_rule == "fixed":
        return (*run_one(ctx, ds, cid, pi, split, seed, subset, n_t, train_ids, test_ids, k_list, gammas, done_fixed, prefilter, save_scores), [])
    todo = [k for k in k_list if (split, subset, n_t, k) not in done_cv]
    if not todo:
        return [], [], [], []
    rows, gene_rows, score_rows = run_one(ctx, ds, cid, pi, split, seed, subset, n_t, train_ids, test_ids, todo, gammas, set(), prefilter, save_scores)
    index = {s: i for i, s in enumerate(ds.sample_ids)}
    tr = np.array([index[s] for s in train_ids])
    n_t_int = 10**6 if n_t == "pool" else int(n_t)
    t0 = time.perf_counter()
    with threadpool_limits(1):
        path = gamma_path(ctx.Xt[tr], ctx.yt[tr], pi, ctx.pre, gammas, todo, prefilter, C.seed_for(cid, seed, subset, n_t_int, GAMMA_SEED_TAG))
    dt = time.perf_counter() - t0
    fixed_by_key = {(r["gamma"], r["k"]): r for r in rows}
    base = {"cohort": ds.name, "split": split, "seed": seed, "subset": subset, "n_t": n_t}
    cv_rows, path_rows = [], []
    for k in sorted({key[1] for key in path}):
        chosen = pick_gamma({g: m["oof_logloss"] for (g, kk), m in path.items() if kk == k})
        path_rows += [{**base, "k": k, "gamma": g, **m, "selected": g == chosen} for (g, kk), m in path.items() if kk == k]
        row = fixed_by_key[(chosen, k)]
        cv_rows.append({**row, "gamma_rule": "m5_cv", "select_time_s": row["select_time_s"] + dt})
    new = lambda r: (split, subset, n_t, r["gamma"], r["k"]) not in done_fixed      # chỉ ghi phần fixed chưa có khi chạy tiếp
    return [r for r in rows if new(r)] + cv_rows, [r for r in gene_rows if new(r)], [r for r in score_rows if new(r)], path_rows


def append_csv(path: Path, rows):
    if rows:
        append_csv_aligned(pd.DataFrame(rows), path)


def load_or_compute_pi(ctx: TransferContext, cid: int, path: Path) -> np.ndarray:
    """π tính một lần cho mỗi cohort; đọc lại khi cùng danh sách gen để resume không phải chạy lại 100 tập con."""
    genes = ctx.genes[ctx.pre]
    if path.exists():
        prev = pd.read_csv(path)
        if prev.gene.astype(str).tolist() == genes.tolist():
            return prev.pi.to_numpy(dtype=float)
    pi, c_q = source_stability(ctx, cid)
    pd.DataFrame({"gene": genes, "pi": pi, "C_q": c_q}).to_csv(path, index=False, encoding="utf-8-sig")
    return pi


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
            ctx = TransferContext(source, ds, args.k, selector="l1_lr", prefilter=args.prefilter, seed=C.seed_for(cid, 0, 0, 7))
            pi_path = out / "rq2_source_pi.csv"
            pi = load_or_compute_pi(ctx, cid, pi_path)
        per_run = out / "rq2_mvp_per_run.csv"
        done, done_cv = set(), set()
        if per_run.exists():
            prev = pd.read_csv(per_run, dtype={"n_t": str})
            done = set(map(tuple, prev.loc[prev.gamma_rule == "fixed", RUN_KEY].itertuples(index=False, name=None)))
            done_cv = set(map(tuple, prev.loc[prev.gamma_rule == "m5_cv", ["split", "subset", "n_t", "k"]].itertuples(index=False, name=None)))
        jobs = [r for r in fewshot_runs(man) if not args.n_t or r[3] in args.n_t]
        print(f"[{name}] MVP: {len(jobs)} lần chạy × {len(args.k)} k × {len(args.gamma)} γ (trừ phần đã có); |G| = {len(ctx.genes)}", flush=True)
        for rows, gene_rows, score_rows, path_rows in Parallel(n_jobs=args.n_jobs, backend="loky", return_as="generator")(
                delayed(run_job)(ctx, ds, cid, pi, s, se, su, nt, tr, te, args.k, args.gamma, done, done_cv, args.prefilter, args.save_scores, args.gamma_rule)
                for s, se, su, nt, tr, te in jobs):
            append_csv(per_run, rows)
            append_csv(out / "rq2_mvp_gamma_path.csv", path_rows)
            append_csv(out / "rq2_mvp_selected_genes.csv", gene_rows)
            append_csv(out / "rq2_mvp_scores.csv", score_rows)
        cfg = {"experiment": "RQ2_mvp", "dataset": name, "role": C.DATASETS.get(name), "source": args.source, "source_sha256": source.files_sha256,
               "n_source": source.n, "gene_space_size": int(len(ctx.genes)), "k_grid": args.k, "gamma_grid": args.gamma, "gamma_rule": args.gamma_rule, "gamma_margin": GAMMA_MARGIN,
               "n_t_levels": args.n_t, "pi_epsilon": PI_EPSILON, "prefilter_p0": args.prefilter, "prefilter_rule": PREFILTER_RULE,
               "source_pi_file": str(pi_path), "source_pi_sha256": sha256(pi_path),
               "fewshot_splits_file": str(path), "fewshot_splits_sha256": sha256(path),
               "n_rows": int(len(pd.read_csv(per_run))) if per_run.exists() else 0, "elapsed_s": time.time() - t_all, "versions": all_versions()}
        write_config(out, prov, cfg, data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv", path,
                                                 source.data_dir / "expression.h5", source.data_dir / "metadata.csv"])
        print(f"[{name}] xong: {cfg['n_rows']} dòng", flush=True)


if __name__ == "__main__":
    main()
