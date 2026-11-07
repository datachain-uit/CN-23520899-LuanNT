"""RQ2: MVP trên đúng các tập con few-shot của E1b và E7 (Review 03/11 Mục 7.3). Mọi γ cố định và γ chọn theo M5.

Thử nghiệm sớm PC-04 (bước 8): --early-tcga chia TCGA theo bệnh nhân 70 % nguồn giả / 30 % đích giả rồi so ba nhánh: MVP (γ theo M5),
target-only (γ = 0) và select-on-source; ghi thêm early_split.csv, fewshot_splits.csv, rq2_early_select_on_source.csv và
rq2_early_summary.csv (mean ± SD theo n_T × k, tỉ lệ panel chứa gen IHC) vào <--out>/tcga_early/. Không dùng để kết luận H2.

Mỗi (split, subset, n_t, k) ghi một dòng gamma_rule = "fixed" cho từng γ (để so γ cố định với CV mà không chạy lại) và, với
--gamma-rule cv (mặc định), hai dòng mang chỉ số của đúng γ đã chọn: gamma_rule = "m5_pos" (cấu hình chính, QĐ49: M5 trên γ > 0) và
"m5_cv" (M5 gốc trên mọi γ; ablation PE-09). Khi tổng hợp phải lọc theo gamma_rule.

Lọc sơ bộ (cột prefilter_space, QĐ51): γ > 0 lọc trên nguồn (ctx.pre, nơi có π_j). γ = 0 chạy hai lần: "target" lọc trên L_T như E1b nên trùng
E1b target-only; "source" lọc trên ctx.pre với w ≡ 1, là nhánh cơ chế để tách MVP − TO = (TO_nguồn − TO) + (MVP − TO_nguồn). Hai quy tắc M5 không
xét nhánh "source".

Nhánh đối chứng (--arms, QĐ51; mặc định đủ ba): "d3" gồm select_on_source và pooled bằng đúng TransferContext và seed của E7, chỉ ở n_T ∈ {5, 10, 20};
"ihc" gồm ihc_2 và ihc_4 bằng đúng hàm và seed của E1b. Các dòng này có gamma trống, gamma_rule = "none". prefilter_space của D3 là "source", của IHC là
"none" (panel cố định, không lọc). pooled để trống exact_k, n_filled vì TransferContext.pooled không trả các đại lượng này và transfer.py giữ nguyên như E7.

select_time_s là thời gian của cả đường k (cùng giá trị cho mọi k của một (γ, prefilter_space)); dòng m5_pos, m5_cv cộng thêm thời gian CV chọn γ.

    python scripts/run_rq2_mvp.py --datasets gse37745 --k 2 3 4 5 10 --gamma 0 0.5 1 2 --n-t 5 10 --save-scores

Đầu ra: results/survey_v2/RQ2_mvp/<cohort>/{rq2_mvp_per_run.csv, rq2_mvp_gamma_path.csv, rq2_mvp_selected_genes.csv, rq2_source_pi.csv, config.json}
và rq2_mvp_scores.csv khi có --save-scores. Resume theo (split, subset, n_t, gamma, prefilter_space, k) với dòng fixed, (split, subset, n_t, k) với
hai dòng quy tắc, (split, subset, n_t, method, k) với nhánh đối chứng.
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
from survey_v2.fewshot import build_fewshot_manifest, fewshot_runs, n_t_sort_key, validate_fewshot_manifest  # noqa: E402
from survey_v2.mvp import EARLY_SEED, EARLY_TARGET_FRACTION, GAMMA_MARGIN, L1_C_GRID_RULE, PI_EPSILON, early_split, gamma_path, pick_gamma, pick_gamma_positive, select_weighted, source_stability  # noqa: E402
from survey_v2.panels import all_fixed_panels, panel_indices  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402
from survey_v2.transfer import TransferContext  # noqa: E402
from scripts.build_fewshot_splits import fewshot_path  # noqa: E402
from scripts.run_e1_classification import append_csv_aligned  # noqa: E402

RUN_KEY = ["split", "subset", "n_t", "gamma", "prefilter_space", "k"]
ARM_KEY = ["split", "subset", "n_t", "method", "k"]
GAMMA_GRID = (0.0, 0.5, 1.0, 2.0)
GAMMA_RULES = ("cv", "fixed")
ARMS = ("mvp", "d3", "ihc")
D3_METHODS = ("select_on_source", "pooled")
D3_N_T = tuple(str(v) for v in C.TRANSFER_N_T)
D3_SEED_TAG = 77               # seed_for(cid, seed, subset, n_t, k, 77): đúng seed của E7
IHC_METHODS = ("ihc_2", "ihc_4")
IHC_SEED_TAG = 90              # seed_for(cid, seed, subset, n_t, 90): đúng seed của panel cố định trong E1b
GAMMA_SEED_TAG = 9             # seed của CV chọn γ: seed_for(cid, seed, subset, n_t, 9)
EARLY_DIR = "tcga_early"
EARLY_N_T = ("5", "10")
EARLY_K = (2, 4, 5, 10)
EARLY_ARMS = ("mvp_m5", "target_only", "select_on_source")
PREFILTER_RULE = ("gamma > 0: top variance on source; gamma = 0 runs twice: prefilter_space = target (top variance on L_T, identical to E1b target-only) "
                  "and prefilter_space = source (ctx.pre with w = 1, mechanism arm, QD51)")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.FEWSHOT_COHORTS))
    ap.add_argument("--source", default=C.SOURCE_DATASET)
    ap.add_argument("--k", nargs="+", type=int, default=None, help="mặc định {2, 3, 4, 5, 10}; với --early-tcga là {2, 4, 5, 10}")
    ap.add_argument("--gamma", nargs="+", type=float, default=list(GAMMA_GRID))
    ap.add_argument("--gamma-rule", choices=GAMMA_RULES, default=GAMMA_RULES[0], help="cv: chọn γ bằng CV trong L_T, ghi m5_pos và m5_cv; fixed: chỉ chạy từng γ")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS), help="nhánh cần chạy (QĐ51); --early-tcga chỉ dùng mvp")
    ap.add_argument("--n-t", nargs="+", default=None, help="mức n_T cần chạy (số hoặc 'pool'); mặc định mọi mức trong bản chia")
    ap.add_argument("--prefilter", type=int, default=C.PREFILTER_P0)
    ap.add_argument("--splits-root", type=Path, default=None)
    ap.add_argument("--save-scores", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", default="RQ2_mvp")
    ap.add_argument("--early-tcga", action="store_true", help="thử nghiệm sớm PC-04 trên TCGA (nguồn giả / đích giả); bỏ qua --datasets")
    args = ap.parse_args(argv)
    if args.early_tcga and (args.gamma_rule != "cv" or 0.0 not in args.gamma):
        ap.error("--early-tcga cần --gamma-rule cv và γ = 0 trong --gamma (nhánh target-only)")
    args.k = args.k or list(EARLY_K if args.early_tcga else C.FEWSHOT_K_GRID)
    if args.early_tcga:
        args.n_t = args.n_t or list(EARLY_N_T)
        args.arms = ["mvp"]         # nhánh select-on-source của thử nghiệm sớm ghi riêng, xem select_on_source_rows
    return args


def run_one(ctx: TransferContext, ds, cid, pi, split, seed, subset, n_t, train_ids, test_ids, k_list, gammas, done, prefilter, save_scores):
    """Một (split, subset, n_t). `pi` theo thứ tự cột của ctx.pre. Không có bước nào đọc nhãn test ngoài evaluate_panel."""
    index = {s: i for i, s in enumerate(ds.sample_ids)}
    tr = np.array([index[s] for s in train_ids]); te = np.array([index[s] for s in test_ids])
    Xtr, ytr, Xte, yte = ctx.Xt[tr], ctx.yt[tr], ctx.Xt[te], ctx.yt[te]
    n_t_int = 10**6 if n_t == "pool" else int(n_t)
    base = {"cohort": ds.name, "split": split, "seed": seed, "subset": subset, "n_t": n_t}
    pi_by_gene = _pi_by_gene(ctx, pi)
    ihc = set(C.IHC_4)
    seed_sel = C.seed_for(cid, seed, subset, n_t_int, C.METHOD_ID["l1_lr"])     # cùng seed với E1b; ghép cặp giữa các γ
    rows, gene_rows, score_rows = [], [], []
    with threadpool_limits(1):
        for gamma in gammas:
            for space in (("target", "source") if gamma == 0 else ("source",)):
                todo = [k for k in k_list if (split, subset, n_t, gamma, space, k) not in done]
                if not todo:
                    continue
                t0 = time.perf_counter()
                if space == "target":
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
                    tag = {"k": k, "method": "mvp", "gamma": gamma}
                    rows.append({**base, **tag, "gamma_rule": "fixed", "prefilter_space": space, "auroc": res.auroc, "bacc": res.bacc,
                                 "mcc": res.mcc, "logloss": res.logloss, "brier": res.brier, "C": res.hyperparam, "threshold": res.threshold,
                                 "n_selected": len(cols), "exact_k": sel.info["exact"][k], "n_filled": sel.info["n_filled"][k],
                                 "n_ihc_in_panel": sum(g in ihc for g in genes), "select_time_s": dt})
                    gene_rows += _gene_rows(base, tag, space, genes, cols, pi_by_gene)
                    if save_scores:
                        score_rows += _score_rows(base, tag, space, test_ids, yte, res)
    return rows, gene_rows, score_rows


def _pi_by_gene(ctx: TransferContext, pi) -> np.ndarray:
    out = np.full(len(ctx.genes), np.nan)
    out[ctx.pre] = pi
    return out


def _gene_rows(base, tag, space, genes, cols, pi_by_gene) -> list:
    return [{**base, **tag, "prefilter_space": space, "rank": r + 1, "gene": g, "pi_source": pi_by_gene[c]} for r, (g, c) in enumerate(zip(genes, cols))]


def _score_rows(base, tag, space, test_ids, yte, res) -> list:
    return [{**base, **tag, "prefilter_space": space, "sample_id": s, "y_true": int(y), "score": float(v)} for s, y, v in zip(test_ids, yte, res.scores_test)]


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
    # γ = 0 của đường γ là target-only (lọc trên L_T), nên hai quy tắc M5 không bao giờ lấy nhánh cơ chế γ = 0/source
    fixed_by_key = {(r["gamma"], r["k"]): r for r in rows if r["gamma"] > 0 or r["prefilter_space"] == "target"}
    base = {"cohort": ds.name, "split": split, "seed": seed, "subset": subset, "n_t": n_t}
    cv_rows, path_rows = [], []
    for k in sorted({key[1] for key in path}):
        logloss = {g: m["oof_logloss"] for (g, kk), m in path.items() if kk == k}
        chosen = {"m5_pos": pick_gamma_positive(logloss), "m5_cv": pick_gamma(logloss)}
        path_rows += [{**base, "k": k, "gamma": g, **m, "selected_m5_pos": g == chosen["m5_pos"], "selected_m5_cv": g == chosen["m5_cv"]}
                      for (g, kk), m in path.items() if kk == k]
        for rule, gamma in chosen.items():
            if gamma is not None:
                row = fixed_by_key[(gamma, k)]
                cv_rows.append({**row, "gamma_rule": rule, "select_time_s": row["select_time_s"] + dt})
    new = lambda r: (split, subset, n_t, r["gamma"], r["prefilter_space"], r["k"]) not in done_fixed      # chỉ ghi phần fixed chưa có khi chạy tiếp
    return [r for r in rows if new(r)] + cv_rows, [r for r in gene_rows if new(r)], [r for r in score_rows if new(r)], path_rows


def run_arms(ctx: TransferContext, ds, cid, pi, split, seed, subset, n_t, train_ids, test_ids, k_list, gammas, done_fixed, done_cv, done_arms, prefilter, save_scores, gamma_rule, arms):
    """Một (split, subset, n_t) cho mọi nhánh được chọn. Trả về (rows, gene_rows, score_rows, path_rows)."""
    rows, gene_rows, score_rows, path_rows = [], [], [], []
    if "mvp" in arms:
        rows, gene_rows, score_rows, path_rows = run_job(ctx, ds, cid, pi, split, seed, subset, n_t, train_ids, test_ids, k_list, gammas, done_fixed, done_cv, prefilter, save_scores, gamma_rule)
    pi_by_gene = _pi_by_gene(ctx, pi)
    for arm, fn in (("d3", d3_rows), ("ihc", ihc_rows)):
        if arm in arms:
            r, g, s = fn(ctx, ds, cid, pi_by_gene, split, seed, subset, n_t, train_ids, test_ids, k_list, done_arms, save_scores)
            rows, gene_rows, score_rows = rows + r, gene_rows + g, score_rows + s
    return rows, gene_rows, score_rows, path_rows


def _arm_row(base, method, k, space, res, n_selected, exact_k, n_filled, genes, dt) -> dict:
    ihc = set(C.IHC_4)
    return {**base, "k": k, "method": method, "gamma": np.nan, "gamma_rule": "none", "prefilter_space": space, "auroc": res.auroc, "bacc": res.bacc,
            "mcc": res.mcc, "logloss": res.logloss, "brier": res.brier, "C": res.hyperparam, "threshold": res.threshold, "n_selected": n_selected,
            "exact_k": exact_k, "n_filled": n_filled, "n_ihc_in_panel": sum(g in ihc for g in genes), "select_time_s": dt}


def d3_rows(ctx: TransferContext, ds, cid, pi_by_gene, split, seed, subset, n_t, train_ids, test_ids, k_list, done, save_scores):
    """select_on_source, pooled bằng đúng hàm và seed của E7 (scripts/run_survey_e7_transfer_naive.py), để qua được cổng bằng nhau."""
    rows, gene_rows, score_rows = [], [], []
    if n_t not in D3_N_T:
        return rows, gene_rows, score_rows
    index = {s: i for i, s in enumerate(ds.sample_ids)}
    tr = np.array([index[s] for s in train_ids]); te = np.array([index[s] for s in test_ids])
    n_t_int = 10**6 if n_t == "pool" else int(n_t)
    base = {"cohort": ds.name, "split": split, "seed": seed, "subset": subset, "n_t": n_t}
    with threadpool_limits(1):
        for k in k_list:
            seed_k = C.seed_for(cid, seed, subset, n_t_int, k, D3_SEED_TAG)
            for method in D3_METHODS:
                if (split, subset, n_t, method, k) in done:
                    continue
                t0 = time.perf_counter()
                if method == "select_on_source":
                    res, cols = ctx.select_on_source(k, tr, te, seed_k), ctx.source_panels[k]
                    exact_k, n_filled = ctx.source_info["exact"][k], ctx.source_info["n_filled"][k]
                else:
                    res, cols = ctx.pooled(k, tr, te, seed_k)
                    exact_k, n_filled = np.nan, np.nan
                dt = time.perf_counter() - t0
                genes = ctx.genes[cols]
                tag = {"k": k, "method": method, "gamma": np.nan}
                rows.append(_arm_row(base, method, k, "source", res, len(cols), exact_k, n_filled, genes, dt))
                gene_rows += _gene_rows(base, tag, "source", genes, cols, pi_by_gene)
                if save_scores:
                    score_rows += _score_rows(base, tag, "source", test_ids, ctx.yt[te], res)
    return rows, gene_rows, score_rows


def ihc_rows(ctx: TransferContext, ds, cid, pi_by_gene, split, seed, subset, n_t, train_ids, test_ids, k_list, done, save_scores):
    """Panel IHC cố định bằng đúng hàm và seed của E1b; k là cỡ panel, không phụ thuộc k_list. Đọc ds.X như E1b."""
    rows, gene_rows, score_rows = [], [], []
    index = {s: i for i, s in enumerate(ds.sample_ids)}
    tr = np.array([index[s] for s in train_ids]); te = np.array([index[s] for s in test_ids])
    n_t_int = 10**6 if n_t == "pool" else int(n_t)
    base = {"cohort": ds.name, "split": split, "seed": seed, "subset": subset, "n_t": n_t}
    panels = all_fixed_panels()
    with threadpool_limits(1):
        for method in IHC_METHODS:
            k = len(panels[method])
            if (split, subset, n_t, method, k) in done:
                continue
            cols, missing = panel_indices(ds.genes, panels[method])
            if len(cols) == 0:
                continue
            t0 = time.perf_counter()
            res = evaluate_panel(ds.X[tr][:, cols], ds.y[tr], ds.X[te][:, cols], ds.y[te], classifier="lr_l2", seed=C.seed_for(cid, seed, subset, n_t_int, IHC_SEED_TAG))
            dt = time.perf_counter() - t0
            genes = ds.genes[cols]
            tag = {"k": k, "method": method, "gamma": np.nan}
            rows.append(_arm_row(base, method, k, "none", res, len(cols), not missing, 0, genes, dt))
            gene_rows += [{**base, **tag, "prefilter_space": "none", "rank": r + 1, "gene": g, "pi_source": np.nan} for r, g in enumerate(genes)]
            if save_scores:
                score_rows += _score_rows(base, tag, "none", test_ids, ds.y[te], res)
    return rows, gene_rows, score_rows


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


def run_target(args, prov, source, ds, out: Path, path: Path, t_all: float, extra_cfg: dict | None = None):
    """Chạy MVP trên một đích theo bản chia `path`, ghi tệp kết quả và config. Trả về (ctx, cid, jobs) để nhánh thử nghiệm sớm dùng lại."""
    name = ds.name
    cid = C.cohort_id(name)
    man = pd.read_csv(path, dtype={"n_t": str})
    errors = validate_fewshot_manifest(man)
    if errors:
        raise ValueError(f"{name}: fewshot_splits.csv không hợp lệ: {errors}")
    with threadpool_limits(1):
        ctx = TransferContext(source, ds, args.k, selector="l1_lr", prefilter=args.prefilter, seed=C.seed_for(cid, 0, 0, 7))
        pi_path = out / "rq2_source_pi.csv"
        pi = load_or_compute_pi(ctx, cid, pi_path)
    per_run = out / "rq2_mvp_per_run.csv"
    done, done_cv, done_arms = set(), set(), set()
    if per_run.exists():
        prev = pd.read_csv(per_run, dtype={"n_t": str})
        done = set(map(tuple, prev.loc[prev.gamma_rule == "fixed", RUN_KEY].itertuples(index=False, name=None)))
        done_arms = set(map(tuple, prev.loc[prev.method != "mvp", ARM_KEY].itertuples(index=False, name=None)))
        done_cv = set(map(tuple, prev.loc[prev.gamma_rule == "m5_cv", ["split", "subset", "n_t", "k"]].itertuples(index=False, name=None)))
    jobs = [r for r in fewshot_runs(man) if not args.n_t or r[3] in args.n_t]
    print(f"[{name}] MVP: {len(jobs)} lần chạy × {len(args.k)} k × {len(args.gamma)} γ (trừ phần đã có); |G| = {len(ctx.genes)}", flush=True)
    for rows, gene_rows, score_rows, path_rows in Parallel(n_jobs=args.n_jobs, backend="loky", return_as="generator")(
            delayed(run_arms)(ctx, ds, cid, pi, s, se, su, nt, tr, te, args.k, args.gamma, done, done_cv, done_arms, args.prefilter, args.save_scores, args.gamma_rule, args.arms)
            for s, se, su, nt, tr, te in jobs):
        append_csv(per_run, rows)
        append_csv(out / "rq2_mvp_gamma_path.csv", path_rows)
        append_csv(out / "rq2_mvp_selected_genes.csv", gene_rows)
        append_csv(out / "rq2_mvp_scores.csv", score_rows)
    cfg = {"experiment": "RQ2_mvp", "dataset": name, "role": C.DATASETS.get(name), "source": args.source, "source_sha256": source.files_sha256,
           "n_source": source.n, "gene_space_size": int(len(ctx.genes)), "k_grid": args.k, "gamma_grid": args.gamma, "gamma_rule": args.gamma_rule, "gamma_margin": GAMMA_MARGIN, "l1_c_grid_rule": L1_C_GRID_RULE, "arms": args.arms,
           "n_t_levels": args.n_t, "pi_epsilon": PI_EPSILON, "prefilter_p0": args.prefilter, "prefilter_rule": PREFILTER_RULE,
           "source_pi_file": str(pi_path), "source_pi_sha256": sha256(pi_path),
           "fewshot_splits_file": str(path), "fewshot_splits_sha256": sha256(path),
           "n_rows": int(len(pd.read_csv(per_run))) if per_run.exists() else 0, "elapsed_s": time.time() - t_all, "versions": all_versions(), **(extra_cfg or {})}
    write_config(out, prov, cfg, data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv", path,
                                             source.data_dir / "expression.h5", source.data_dir / "metadata.csv"])
    print(f"[{name}] xong: {cfg['n_rows']} dòng", flush=True)
    return ctx, cid, jobs


def select_on_source_rows(ctx: TransferContext, ds, cid, jobs, k_list, done) -> list:
    """Nhánh select-on-source: panel chọn trên nguồn giả, LR-L2 fit trên L_T. Cùng tập con và seed với MVP để ghép cặp."""
    index = {s: i for i, s in enumerate(ds.sample_ids)}
    ihc = set(C.IHC_4)
    rows = []
    with threadpool_limits(1):
        for split, seed, subset, n_t, train_ids, test_ids in jobs:
            tr = np.array([index[s] for s in train_ids]); te = np.array([index[s] for s in test_ids])
            n_t_int = 10**6 if n_t == "pool" else int(n_t)
            for k in k_list:
                if (split, subset, n_t, k) in done:
                    continue
                res = ctx.select_on_source(k, tr, te, seed=C.seed_for(cid, seed, subset, n_t_int, k))
                panel = ctx.source_panels[k]
                rows.append({"cohort": ds.name, "split": split, "seed": seed, "subset": subset, "n_t": n_t, "k": k, "method": "select_on_source",
                             "gamma": np.nan, "gamma_rule": "none", "auroc": res.auroc, "bacc": res.bacc, "mcc": res.mcc, "logloss": res.logloss,
                             "brier": res.brier, "C": res.hyperparam, "threshold": res.threshold, "n_selected": len(panel),
                             "exact_k": ctx.source_info["exact"][k], "n_filled": ctx.source_info["n_filled"][k],
                             "n_ihc_in_panel": sum(g in ihc for g in ctx.genes[panel]), "select_time_s": 0.0})
    return rows


def early_summary(per_run: pd.DataFrame, select_on_source: pd.DataFrame) -> pd.DataFrame:
    """Bảng mean ± SD theo n_T × k cho ba nhánh; ihc_panel_rate = tỉ lệ lần chạy có panel chứa ít nhất một gen IHC-4."""
    frames = {"mvp_m5": per_run[per_run.gamma_rule == "m5_cv"],
              "target_only": per_run[(per_run.gamma_rule == "fixed") & (per_run.gamma == 0) & (per_run.prefilter_space == "target")],
              "select_on_source": select_on_source}
    rows = []
    for arm in EARLY_ARMS:
        for (n_t, k), g in frames[arm].groupby(["n_t", "k"]):
            rows.append({"n_t": n_t, "k": int(k), "arm": arm, "n_runs": len(g), "auroc_mean": g.auroc.mean(), "auroc_sd": g.auroc.std(),
                         "bacc_mean": g.bacc.mean(), "bacc_sd": g.bacc.std(), "logloss_mean": g.logloss.mean(), "logloss_sd": g.logloss.std(),
                         "ihc_panel_rate": float((g.n_ihc_in_panel > 0).mean())})
    out = pd.DataFrame(rows)
    out["_n"], out["_a"] = out.n_t.map(n_t_sort_key), out.arm.map(EARLY_ARMS.index)
    return out.sort_values(["_n", "k", "_a"]).drop(columns=["_n", "_a"]).reset_index(drop=True)


def run_early(args, prov, t_all: float) -> None:
    full = load_dataset(DATA_ROOT, args.source, gene_space=None)
    source, target = early_split(full)
    out = results_dir(args.out, EARLY_DIR)
    split_path = out / "early_split.csv"
    pd.concat([pd.DataFrame({"sample_id": d.sample_ids, "patient_id": d.groups, "label": d.y, "role": role})
               for d, role in ((source, "source"), (target, "target"))]).to_csv(split_path, index=False, encoding="utf-8-sig")
    man = build_fewshot_manifest(target.name, target.sample_ids, target.groups, target.y)
    path = out / "fewshot_splits.csv"
    man[man.n_t.isin(args.n_t)].to_csv(path, index=False, encoding="utf-8-sig")
    extra = {"experiment": "RQ2_mvp_early", "dataset": target.name, "early_seed": EARLY_SEED, "early_target_fraction": EARLY_TARGET_FRACTION,
             "n_target": target.n, "early_split_sha256": sha256(split_path), "note": "thử nghiệm sớm PC-04; không kết luận H2"}
    ctx, cid, jobs = run_target(args, prov, source, target, out, path, t_all, extra)
    sos_path = out / "rq2_early_select_on_source.csv"
    done = set()
    if sos_path.exists():
        prev = pd.read_csv(sos_path, dtype={"n_t": str})
        done = set(map(tuple, prev[["split", "subset", "n_t", "k"]].itertuples(index=False, name=None)))
    append_csv(sos_path, select_on_source_rows(ctx, target, cid, jobs, args.k, done))
    summary = early_summary(pd.read_csv(out / "rq2_mvp_per_run.csv", dtype={"n_t": str}), pd.read_csv(sos_path, dtype={"n_t": str}))
    summary.to_csv(out / "rq2_early_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False), flush=True)


def main(argv=None, script=__file__):
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t_all = time.time()
    if args.early_tcga:
        run_early(args, prov, t_all)
        return
    source = load_dataset(DATA_ROOT, args.source, gene_space=None)
    for name in args.datasets:
        ds, out = load_dataset(DATA_ROOT, name), results_dir(args.out, name)
        run_target(args, prov, source, ds, out, fewshot_path(ds, args.splits_root), t_all)


if __name__ == "__main__":
    main()
