"""Panel và đánh giá baseline R của RQ2 (QĐ68, Review 08/11 Mục 8.3 Bảng 8 bước 5, 6, 7, 8) từ hệ số do scripts/r/rq2_r_baselines.R ghi.

ptLasso:
  * chọn α: LL(α) = cvm(λ2)/2 (deviance CV chia 2 = log-loss CV); xét α giảm dần từ 0,75, α nhỏ hơn thay α đang giữ chỉ khi
    LL giảm ≥ 0,01 (cùng biên với m5_pos); α lỗi bị bỏ qua; một α cho mọi k;
  * β_eff = (1 − α̂)·β1 + β2(λ2) trên 5.000 gen; panel = k gen có |β_eff| > 0 lớn nhất (hoà theo thứ tự trong danh sách 5.000 gen);
    thiếu thì bổ sung F-test trên L_T (n_filled);
  * cờ same_as_stage1_topk = panel trùng tập k gen có |β1| lớn nhất (NaN khi |S1| < k).
Trans-GLM: panel = k gen có |β| > 0 lớn nhất (bỏ hệ số chặn), hoà theo thứ tự, thiếu thì F-test.
Đánh giá: evaluate_panel (LR-L2, M5) với seed_for(cid, seed, subset, n_t, k) như MVP; rq2_r_per_run.csv cùng cột với rq2_mvp_per_run.csv
(method = ptlasso, transglm), điểm từng mẫu trong rq2_r_scores.csv.

    python scripts/run_rq2_r_eval.py --inputs <RQ2_r_inputs> --r-out <thư mục kết quả R> --save-scores

Đầu ra: results/survey_v2/RQ2_r/<cohort>/{rq2_r_per_run.csv, rq2_r_selected_genes.csv, rq2_r_diagnostics.csv, rq2_r_scores.csv, config.json}.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.classify import evaluate_panel  # noqa: E402
from survey_v2.data import load_dataset, sha256  # noqa: E402
from survey_v2.methods import ftest_order  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402

ALPHAS = (0.75, 0.5, 0.25)
ALPHA_MARGIN = 0.01
# QĐ70: đánh giá phải cùng môi trường với MVP để AUROC so được theo cặp. Giá trị = trường versions của RQ2_mvp/*/config.json
# (rq2_mvp_1b7c021_results.zip), trùng E1b, E7 (9b2d6a5, d870008). Lệch thì dừng mã 2 trước khi tính gì.
REFERENCE_VERSIONS = {"numpy": "2.1.3", "pandas": "2.3.3", "scipy": "1.16.3", "scikit-learn": "1.6.1"}
METHODS = ("ptlasso", "transglm")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", type=Path, required=True, help="thư mục của scripts/export_rq2_inputs.py")
    ap.add_argument("--r-out", type=Path, required=True, help="thư mục kết quả của scripts/r/rq2_r_baselines.R")
    ap.add_argument("--datasets", nargs="+", default=list(C.FEWSHOT_COHORTS))
    ap.add_argument("--k", nargs="+", type=int, default=list(C.FEWSHOT_K_GRID))
    ap.add_argument("--save-scores", action="store_true")
    ap.add_argument("--out", default="RQ2_r")
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------------------------- quy tắc
def choose_alpha(rows: pd.DataFrame, margin: float = ALPHA_MARGIN) -> tuple[float | None, dict]:
    """rows: các dòng ptlasso_runs.csv của một lần chạy. Trả (α̂ hoặc None, {α: LL})."""
    ll = {float(r.alpha): float(r.cvm_min) / 2 for r in rows.itertuples() if pd.isna(r.error) and np.isfinite(r.cvm_min)}
    best = None
    for a in sorted(ll, reverse=True):
        if best is None or ll[a] <= ll[best] - margin:
            best = a
    return best, ll


def top_k(values: np.ndarray, k: int) -> list:
    """Chỉ số của tối đa k phần tử có |giá trị| > 0 lớn nhất; hoà theo chỉ số (thứ tự trong danh sách 5.000 gen)."""
    v = np.abs(np.asarray(values, float))
    nz = np.flatnonzero(v > 0)
    order = nz[np.argsort(-v[nz], kind="stable")]
    return order[:k].tolist()


def panel_with_fill(values: np.ndarray, k: int, X_train: np.ndarray, y_train: np.ndarray) -> tuple[list, int]:
    """(chỉ số panel trong 5.000 gen theo thứ tự hạng, n_filled). Thiếu k gen khác 0 thì bổ sung theo F-test trên L_T."""
    chosen = top_k(values, k)
    n_filled = 0
    if len(chosen) < k:
        taken = set(chosen)
        for j in ftest_order(X_train, y_train):
            if len(chosen) == k:
                break
            if int(j) not in taken:
                chosen.append(int(j)); taken.add(int(j)); n_filled += 1
    return chosen, n_filled


def same_as_stage1_topk(beta1: np.ndarray, panel: list, k: int) -> float:
    if int(np.sum(np.asarray(beta1) != 0)) < k:
        return float("nan")
    return float(set(top_k(beta1, k)) == set(panel))


# ------------------------------------------------------------------------------------------------- đọc dữ liệu
def _beta_vector(frame: pd.DataFrame, genes: list, value: str) -> np.ndarray:
    idx = {g: j for j, g in enumerate(genes)}
    v = np.zeros(len(genes))
    for g, b in zip(frame.gene, frame[value]):
        v[idx[g]] = b
    return v


def load_cohort(inputs: Path, r_out: Path, cohort: str) -> dict:
    space = (inputs / cohort / "space.txt").read_text(encoding="utf-8").strip()
    genes = pd.read_csv(inputs / space / "genes.csv").gene.astype(str).tolist()
    s1 = pd.read_csv(r_out / f"stage1_{space}.csv")
    s1 = s1[s1.gene != "(Intercept)"]
    if s1.gene.astype(str).tolist() != genes:
        raise ValueError(f"{cohort}: thứ tự gen của giai đoạn 1 khác genes.csv")
    def read(name):                                   # R ghi bảng rỗng (không lần chạy nào có hệ số khác 0) thành tệp không có cột
        try:
            return pd.read_csv(r_out / cohort / name, dtype={"run_id": str, "gene": str})
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    return {"space": space, "genes": genes, "beta1": s1.beta1.to_numpy(float),
            "runs": pd.read_csv(inputs / cohort / "runs.csv", dtype={"run_id": str, "n_t": str}),
            "members": pd.read_csv(inputs / cohort / "membership.csv", dtype={"run_id": str, "sample_id": str}),
            "pt_runs": read("ptlasso_runs.csv"), "pt_beta": read("ptlasso_beta2.csv"),
            "tg_runs": read("transglm_runs.csv"), "tg_beta": read("transglm_beta.csv")}


# ------------------------------------------------------------------------------------------------- đánh giá
def evaluate_cohort(ds, data: dict, k_list, save_scores: bool) -> tuple[list, list, list, list]:
    """(dòng per_run, dòng gen, dòng chẩn đoán, dòng điểm)."""
    genes, beta1 = data["genes"], data["beta1"]
    gi = {g: j for j, g in enumerate(ds.genes)}
    cols_pre = np.array([gi[g] for g in genes])
    index = {s: i for i, s in enumerate(ds.sample_ids)}
    cid = C.cohort_id(ds.name)
    ihc = set(C.IHC_4)
    pt_runs = data["pt_runs"].groupby("run_id") if len(data["pt_runs"]) else {}
    pt_beta = data["pt_beta"]
    tg_runs = data["tg_runs"].set_index("run_id") if len(data["tg_runs"]) else pd.DataFrame()
    tg_beta = data["tg_beta"]
    rows, gene_rows, diag_rows, score_rows = [], [], [], []
    for run in data["runs"].itertuples():
        m = data["members"][data["members"].run_id == run.run_id]
        tr = np.array([index[s] for s in m.loc[m.role == "train", "sample_id"]])
        te_ids = m.loc[m.role == "test", "sample_id"].to_numpy(str)
        te = np.array([index[s] for s in te_ids])
        Xtr_pre, ytr = ds.X[tr][:, cols_pre], ds.y[tr]
        n_t_int = 10**6 if run.n_t == "pool" else int(run.n_t)
        base = {"cohort": ds.name, "split": int(run.split), "seed": int(run.seed), "subset": int(run.subset), "n_t": str(run.n_t)}
        # --- hệ số của từng phương pháp
        coefs, diag = {}, {}
        rr = pt_runs.get_group(run.run_id) if hasattr(pt_runs, "get_group") and run.run_id in pt_runs.groups else pd.DataFrame(columns=["alpha", "cvm_min", "error"])
        alpha, ll = choose_alpha(rr)
        if alpha is not None:
            b2 = pt_beta[(pt_beta.run_id == run.run_id) & (pt_beta.alpha == alpha)] if len(pt_beta) else pt_beta
            coefs["ptlasso"] = (1 - alpha) * beta1 + _beta_vector(b2, genes, "beta2")
            sel = rr[rr.alpha == alpha].iloc[0]
            diag["ptlasso"] = {"alpha": alpha, **{f"logloss_alpha_{a:g}": ll.get(a, np.nan) for a in ALPHAS}, "lambda_min": float(sel.lambda_min),
                               "nnz_beta2": int(sel.nnz_beta2), "size_S1": int(np.sum(beta1 != 0)), "seconds": float(rr.seconds.sum())}
        else:
            diag["ptlasso"] = {"error": "; ".join(rr.error.dropna().astype(str).unique()) or "không có α nào", "size_S1": int(np.sum(beta1 != 0))}
        if len(tg_runs) and run.run_id in tg_runs.index and pd.isna(tg_runs.loc[run.run_id, "error"]):
            t = tg_runs.loc[run.run_id]
            coefs["transglm"] = _beta_vector(tg_beta[tg_beta.run_id == run.run_id] if len(tg_beta) else tg_beta, genes, "beta")
            diag["transglm"] = {"transfer_source_id": t.transfer_source_id, "nnz": int(t.nnz), "seconds": float(t.seconds)}
        else:
            diag["transglm"] = {"error": str(tg_runs.loc[run.run_id, "error"]) if len(tg_runs) and run.run_id in tg_runs.index else "không có kết quả R"}
        # --- panel và đánh giá
        for method in METHODS:
            for k in sorted(int(x) for x in k_list):
                d = {**base, "method": method, "k": k, **diag[method]}
                if method not in coefs:
                    diag_rows.append(d)
                    continue
                panel, n_filled = panel_with_fill(coefs[method], k, Xtr_pre, ytr)
                if method == "ptlasso":
                    d["same_as_stage1_topk"] = same_as_stage1_topk(beta1, panel, k)
                diag_rows.append({**d, "n_filled": n_filled})
                cols = cols_pre[panel]
                pg = [genes[j] for j in panel]
                res = evaluate_panel(ds.X[tr][:, cols], ytr, ds.X[te][:, cols], ds.y[te], classifier="lr_l2", seed=C.seed_for(cid, int(run.seed), int(run.subset), n_t_int, k))
                rows.append({**base, "k": k, "method": method, "gamma": np.nan, "gamma_rule": "none", "prefilter_space": "source", "auroc": res.auroc,
                             "bacc": res.bacc, "mcc": res.mcc, "logloss": res.logloss, "brier": res.brier, "C": res.hyperparam, "threshold": res.threshold,
                             "n_selected": len(cols), "exact_k": n_filled == 0, "n_filled": n_filled, "n_ihc_in_panel": sum(g in ihc for g in pg),
                             "select_time_s": diag[method].get("seconds", np.nan)})
                gene_rows += [{**base, "k": k, "method": method, "gamma": np.nan, "prefilter_space": "source", "rank": r + 1, "gene": g,
                               "coef": float(coefs[method][panel[r]]), "filled": r >= len(panel) - n_filled} for r, g in enumerate(pg)]
                if save_scores:
                    score_rows += [{**base, "k": k, "method": method, "gamma": np.nan, "prefilter_space": "source", "sample_id": s, "y_true": int(y),
                                    "score": float(v)} for s, y, v in zip(te_ids, ds.y[te], res.scores_test)]
    return rows, gene_rows, diag_rows, score_rows


def version_mismatches(expected: dict = REFERENCE_VERSIONS, current: dict | None = None) -> dict:
    """{gói: (phiên bản đang có, phiên bản tham chiếu)} cho các gói lệch; rỗng khi khớp."""
    from importlib import metadata
    if current is None:
        current = {}
        for p in expected:
            try:
                current[p] = metadata.version(p)
            except metadata.PackageNotFoundError:
                current[p] = None
    return {p: (current.get(p), v) for p, v in expected.items() if current.get(p) != v}


def main(argv=None, script=__file__) -> int:
    args = parse_args(argv)
    bad = version_mismatches()
    if bad:
        print("PHIÊN BẢN THƯ VIỆN KHÁC rq2_mvp (QĐ70) — không đánh giá:\n  " + "\n  ".join(f"{p}: phiên {c}, tham chiếu {v}" for p, (c, v) in bad.items())
              + "\ncài đúng phiên bản rồi chạy lại — notebook kaggle_rq2_r_run: dán vào ô cấu hình\n  PIP_PINS = " + repr({p: v for p, (_, v) in bad.items()}))
        return 2
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    stage1 = pd.read_csv(args.r_out / "stage1_summary.csv")
    session = args.r_out / "sessionInfo.txt"
    for name in args.datasets:
        t0 = time.time()
        ds = load_dataset(DATA_ROOT, name)
        data = load_cohort(args.inputs, args.r_out, name)
        rows, gene_rows, diag_rows, score_rows = evaluate_cohort(ds, data, args.k, args.save_scores)
        out = results_dir(args.out, name)
        pd.DataFrame(rows).to_csv(out / "rq2_r_per_run.csv", index=False)
        pd.DataFrame(gene_rows).to_csv(out / "rq2_r_selected_genes.csv", index=False)
        diag = pd.DataFrame(diag_rows)
        diag.to_csv(out / "rq2_r_diagnostics.csv", index=False)
        if args.save_scores:
            pd.DataFrame(score_rows).to_csv(out / "rq2_r_scores.csv", index=False)
        pt = diag[diag.method == "ptlasso"]
        cfg = {"experiment": "RQ2_r", "dataset": name, "space": data["space"], "k_grid": args.k, "methods": list(METHODS), "alphas": list(ALPHAS),
               "alpha_margin": ALPHA_MARGIN, "stage1": stage1[stage1.space == data["space"]].to_dict("records"),
               "n_runs": int(len(data["runs"])), "n_rows": len(rows), "n_errors": int(diag["error"].notna().sum()) if "error" in diag else 0,
               "same_as_stage1_topk_rate": float(pt.same_as_stage1_topk.mean()) if "same_as_stage1_topk" in pt and pt.same_as_stage1_topk.notna().any() else None,
               "r_session_info": session.read_text(encoding="utf-8", errors="replace").splitlines()[:40] if session.exists() else None,
               "panel_rule": "k gen |β| > 0 lớn nhất, hoà theo thứ tự 5.000 gen, thiếu thì F-test trên L_T; ptLasso β_eff = (1 − α̂)β1 + β2",
               "evaluation": "evaluate_panel lr_l2, seed_for(cid, seed, subset, n_t, k) như MVP", "elapsed_s": time.time() - t0, "versions": all_versions()}
        write_config(out, prov, cfg, data_files=[args.r_out / "stage1_summary.csv"])
        print(f"[{name}] {len(rows)} dòng, lỗi {cfg['n_errors']}, same_as_stage1_topk {cfg['same_as_stage1_topk_rate']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
