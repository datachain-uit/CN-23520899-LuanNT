"""Tổng hợp E0–E7 thành bảng (CSV + LaTeX booktabs) và hình cho bản thảo survey v3; kèm manifest truy vết.

Đọc results/survey_v2/{E0, E1_classification, E1_feature_selection, E1b, E2_E3, E4_E5, E7}/... (phần nào chưa có thì bỏ qua
và báo), ghi vào results/survey_v2/tables/ và results/survey_v2/figures/. Mọi bảng ghi kèm `provenance_manifest.json`:
tệp đầu vào, SHA-256, commit hash của từng config.json — người viết trích số liệu từ đây và ghi [source: ...] vào
working/03_claims.md.

Thống kê (KBTN v2.1; QĐ8, QĐ11, QĐ13):
* E1: mean, SD và CI 95% (phương sai Nadeau–Bengio) qua J = 25 fold (wrapper J = 15, ghép cặp theo (repeat, fold)), từ
  pha B (classification_per_fold.csv, model = --classifier).
* Kiểm định vs random-k: d_j = AUROC(method, k, fold j) − trung bình các lần rút random-k trên fold j; t-test Nadeau–Bengio
  một phía; Holm trong họ = một phương pháp × các mức k; thêm p_random_empirical (phân bố trung bình-qua-fold của từng lần rút).
* Kiểm định vs B0: hai phía; Holm trong cùng họ.
* k* gộp qua 3 cohort GPL570 cho MỌI phương pháp (survey_v2/kstar.py = đoạn code G): IVW + trọng số bằng nhau, chuỗi 50 → 1;
  kèm k* từng cohort (mô tả).
* Bayesian correlated t-test (ROPE ±0,01) cho cặp dẫn đầu ở k ∈ ROPE_K trên từng cohort GPL570; ma trận đầy đủ ở k = OVERLAP_K
  trên RANKING_DATASET.
* E1b: trung bình ± SD qua 20 lần chạy, CI bootstrap qua các lần chạy; n_T* (mức n_T nhỏ nhất trong Δ = 0,02 so với pool).
* E7: chênh lệch ghép cặp (chiến lược − target_only) với CI bootstrap qua các lần chạy; Holm trên 3 mức n_T.
* E6: ghép AUROC (E1, k = OVERLAP_K) – Nogueira (E4, tỉ lệ --e6-fraction) – thời gian (E2, k = OVERLAP_K); mặt Pareto.

    python scripts/make_survey_tables.py
    python scripts/make_survey_tables.py --e6-fraction 0.9 --no-figures
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import sha256  # noqa: E402
from survey_v2.runlog import DATA_ROOT, results_dir, start_run, write_config  # noqa: E402
from survey_v2.kstar import kstar_table, per_cohort_kstar  # noqa: E402
from survey_v2.stats import bayesian_correlated_ttest, holm, mean_ci, nadeau_bengio_ttest  # noqa: E402

METHOD_LABEL = {"f_test": "F-test", "mutual_info": "MI", "mrmr": "mRMR", "relieff": "ReliefF", "l1_lr": "L1-LR",
                "elastic_net": "Elastic Net", "rf_importance": "RF importance", "svm_rfe": "SVM-RFE", "abess": "abess (L0)",
                "stability_selection": "Stability selection", "ktsp": "k-TSP", "ga": "GA", "bpso": "BPSO",
                "b0": "B0 (all genes)", "random_k": "random-k", "ihc_2": "IHC-2", "ihc_4": "IHC-4"}
METHOD_ORDER = list(C.ALL_SELECTORS) + ["ihc_2", "ihc_4", "random_k", "b0"]
REFERENCE_SET = {"b0", "random_k", "ihc_2", "ihc_4"}


def is_reference(m: str) -> bool:
    return m in REFERENCE_SET or str(m).startswith("panel_")
# Màu theo HỌ (3 màu đã kiểm tra an toàn mù màu theo mọi cặp), tham chiếu màu trung tính; phương pháp phân biệt
# bằng marker + kiểu nét + nhãn trực tiếp (mã hóa kép, không dựa vào màu đơn thuần).
FAMILY_COLOR = {"filter": "#2a78d6", "embedded": "#eb6834", "constrained": "#8e44ad", "rank-based": "#8e44ad",
                "wrapper": "#1baf7a", "reference": "#6b6b6b", "knowledge": "#b5651d", "other": "#6b6b6b"}
METHOD_MARKER = {"f_test": "o", "mutual_info": "s", "mrmr": "^", "relieff": "D", "l1_lr": "o", "elastic_net": "s",
                 "rf_importance": "^", "svm_rfe": "D", "abess": "o", "stability_selection": "s", "ktsp": "^",
                 "ga": "o", "bpso": "s", "b0": "x", "random_k": "+", "ihc_2": "*", "ihc_4": "P"}
METHOD_LS = {"f_test": "-", "mutual_info": "--", "mrmr": "-.", "relieff": ":", "l1_lr": "-", "elastic_net": "--",
             "rf_importance": "-.", "svm_rfe": ":", "abess": "-", "stability_selection": "--", "ktsp": "-.",
             "ga": "-", "bpso": "--", "b0": "-", "random_k": "--", "ihc_2": ":", "ihc_4": ":"}
# Số mũ lý thuyết theo Bảng 5 của bản v1 (draft/05_time_complexity.tex, tab:fs_complexity), quy về (n, p) với k cố định.
TABLE5_EXPONENTS = {
    "f_test": (1, 1, r"O(np)"), "mutual_info": (1, 1, r"O(np)"), "relieff": (2, 1, r"O(n^2 p)"),
    "mrmr": (1, 2, r"O(p^2 n) [cài đặt tham lam k gen: O(k n p) -> kỳ vọng alpha_p = 1]"),
    "l1_lr": (1, 1, r"O(n p t)"), "elastic_net": (1, 1, r"O(n p t)"), "rf_importance": (1, 1, r"O(B n p log n)"),
    "svm_rfe": (2, 2, r"O(n^2 p^2) [loại 10%/vòng: ~O(n^2 p) hoặc O(n p^2) tùy n, p]"),
    "ga": (None, None, r"O(G·C(k))"), "bpso": (None, None, r"O(I·C(k))"),
    "abess": (1, 1, r"O(n p · iterations) [splicing; Zhu et al. 2020]"), "stability_selection": (1, 1, r"O(B · n p t) [B = 100 L1 fits]"),
    "ktsp": (1, 2, r"O(n p^2) [pair scores on the pre-filtered genes]"),
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e1-dir", default="E1_classification", help="thư mục pha B của E1")
    ap.add_argument("--e1-select-dir", default="E1_feature_selection")
    ap.add_argument("--e0-dir", default="E0")
    ap.add_argument("--e1b-dir", default="E1b")
    ap.add_argument("--e7-dir", default="E7")
    ap.add_argument("--e2e3-dir", default="E2_E3")
    ap.add_argument("--e4e5-dir", default="E4_E5")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--fig-out", default="figures")
    ap.add_argument("--classifier", default="lr_l2")
    ap.add_argument("--e6-fraction", type=float, default=C.REDUNDANCY_FRACTION)
    ap.add_argument("--no-figures", action="store_true")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------------------------------
class Manifest:
    def __init__(self):
        self.inputs: dict[str, dict] = {}
        self.outputs: list[str] = []

    def read_csv(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        cfg = path.parent / "config.json"
        commit = None
        if cfg.exists():
            try:
                commit = json.loads(cfg.read_text(encoding="utf-8")).get("commit_hash")
            except Exception:
                commit = None
        self.inputs[str(path)] = {"sha256": sha256(path), "rows": int(len(df)), "commit_hash": commit}
        return df

    def save(self, out_dir: Path, prov: dict):
        payload = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "script_commit": prov.get("commit_hash"),
                   "inputs": self.inputs, "outputs": sorted(set(self.outputs)),
                   "input_commit_hashes": sorted({str(v["commit_hash"]) for v in self.inputs.values()})}
        (out_dir / "provenance_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fmt_pm(m, s, nd=3):
    return "" if pd.isna(m) else (f"{m:.{nd}f}" if pd.isna(s) else f"{m:.{nd}f} ± {s:.{nd}f}")


def to_booktabs(df: pd.DataFrame, path: Path, caption: str, label: str, float_fmt: str = "{:.3f}") -> None:
    """Ghi bảng LaTeX booktabs đơn giản (không phụ thuộc DataFrame.to_latex)."""
    def cell(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else float_fmt.format(v)
        return str(v).replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$")
    cols = [cell(c) for c in df.columns]
    lines = [r"\begin{table*}[t]", r"  \centering", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}", r"  \small",
             "  \\begin{tabular}{" + "l" * len(cols) + "}", r"    \toprule", "    " + " & ".join(cols) + r" \\", r"    \midrule"]
    for row in df.itertuples(index=False):
        lines.append("    " + " & ".join(cell(v) for v in row) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def save(df: pd.DataFrame, out: Path, name: str, man: Manifest, tex: tuple | None = None):
    p = out / f"{name}.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    man.outputs.append(str(p))
    if tex is not None:
        caption, label = tex
        to_booktabs(df, out / f"{name}.tex", caption, label)
        man.outputs.append(str(out / f"{name}.tex"))
    print(f"  ghi {name}: {len(df)} dòng", flush=True)


# ---------------------------------------------------------------------------------------------------
# E1
# ---------------------------------------------------------------------------------------------------
def e1_summary(per_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (co, m, k, clf), g in per_fold.groupby(["cohort", "method", "k", "classifier"]):
        row = {"cohort": co, "method": m, "k": int(k), "classifier": clf, "n_folds": len(g)}
        for metric in ("auroc", "bacc", "mcc", "logloss", "brier"):
            if metric not in g.columns:
                continue
            s = mean_ci(g[metric].to_numpy())
            row.update({f"{metric}_mean": s["mean"], f"{metric}_sd": s["sd"], f"{metric}_ci_low": s["ci_low"],
                        f"{metric}_ci_high": s["ci_high"]})
        for extra in ("select_time_s", "n_filled", "n_eval", "best_fitness", "fit_time_s"):
            if extra in g.columns:
                row[f"{extra}_mean"] = float(pd.to_numeric(g[extra], errors="coerce").mean())
        if "exact_k" in g.columns:
            row["exact_k_rate"] = float(g["exact_k"].astype(str).str.lower().eq("true").mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["method_order"] = df["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
    return df.sort_values(["cohort", "classifier", "k", "method_order"]).drop(columns="method_order")


def load_phase_b(e1_root: Path, select_root: Path, man: "Manifest"):
    """Ghép pha B (chỉ số) với pha A (select_time_s, n_filled, exact_k, n_eval) theo (cohort, repeat, fold, method, k)."""
    per_fold_all, random_all = [], []
    if not e1_root.exists():
        return None, None
    for d in sorted(p for p in e1_root.iterdir() if p.is_dir()):
        pf = man.read_csv(d / "classification_per_fold.csv")
        if pf is None:
            continue
        pf = pf.rename(columns={"model": "classifier"})
        sel = man.read_csv(select_root / d.name / "selection_per_fold.csv") if (select_root / d.name / "selection_per_fold.csv").exists() else None
        if sel is not None and len(sel):
            keep = [c for c in ("cohort", "repeat", "fold", "method", "k", "selection_time_s", "n_filled", "exact_k", "n_eval", "best_fitness") if c in sel.columns]
            sel = sel[keep].rename(columns={"selection_time_s": "select_time_s"}).drop_duplicates(["cohort", "repeat", "fold", "method", "k"])
            pf = pf.merge(sel, on=["cohort", "repeat", "fold", "method", "k"], how="left")
        per_fold_all.append(pf)
        rk = man.read_csv(d / "e1_random_k.csv")
        if rk is not None:
            random_all.append(rk)
    if not per_fold_all:
        return None, None
    return pd.concat(per_fold_all, ignore_index=True), (pd.concat(random_all, ignore_index=True) if random_all else None)


def random_per_fold(random_k: pd.DataFrame) -> pd.DataFrame:
    """Kỳ vọng random-k trên từng fold: trung bình (và SD) qua các lần rút; kèm số lần rút."""
    g = random_k.groupby(["cohort", "repeat", "fold", "k"], as_index=False).agg(
        random_auroc_mean=("auroc", "mean"), random_auroc_sd=("auroc", "std"), random_auroc_q95=("auroc", lambda s: s.quantile(0.95)),
        random_bacc_mean=("bacc", "mean"), n_draws=("auroc", "size"))
    return g


def e1_tests(per_fold: pd.DataFrame, random_k: pd.DataFrame | None, classifier: str) -> pd.DataFrame:
    pf = per_fold[per_fold.classifier == classifier]
    methods = [m for m in pf.method.unique() if m not in ("b0", "random_k")]   # panel tri thức vẫn được so với random-k và B0
    b0 = pf[pf.method == "b0"][["cohort", "repeat", "fold", "auroc"]].rename(columns={"auroc": "auroc_b0"})
    rnd = random_per_fold(random_k) if random_k is not None and len(random_k) else None
    rows = []
    for co in pf.cohort.unique():
        for m in methods:
            for k, g in pf[(pf.cohort == co) & (pf.method == m)].groupby("k"):
                base = {"cohort": co, "method": m, "k": int(k), "classifier": classifier, "n_folds": len(g),
                        "auroc_mean": float(g.auroc.mean())}
                if rnd is not None:
                    j = g.merge(rnd[(rnd.cohort == co) & (rnd.k == k)], on=["cohort", "repeat", "fold", "k"], how="inner")
                    if len(j) >= 2:
                        t = nadeau_bengio_ttest((j.auroc - j.random_auroc_mean).to_numpy(), alternative="greater")
                        # p kiểu P0: phân bố trung bình-qua-fold của từng lần rút random-k so với trung bình của phương pháp
                        rk = random_k[(random_k.cohort == co) & (random_k.k == k)]
                        rk = rk.merge(g[["repeat", "fold"]].drop_duplicates(), on=["repeat", "fold"])
                        draw_means = rk.groupby("draw")["auroc"].mean().to_numpy()
                        p_emp = (np.sum(draw_means >= j.auroc.mean()) + 1) / (len(draw_means) + 1) if len(draw_means) else np.nan
                        base.update({"random_mean": float(j.random_auroc_mean.mean()), "gain_vs_random": t["mean"],
                                     "gain_ci_low": t["ci_low"], "gain_ci_high": t["ci_high"], "t_nb_random": t["t"],
                                     "p_nb_random": t["p"], "n_pairs_random": t["n"], "p_random_empirical": p_emp,
                                     "n_random_draws": int(len(draw_means))})
                if len(b0):
                    j = g.merge(b0[b0.cohort == co], on=["cohort", "repeat", "fold"], how="inner")
                    if len(j) >= 2:
                        t = nadeau_bengio_ttest((j.auroc - j.auroc_b0).to_numpy(), alternative="two-sided")
                        base.update({"b0_mean": float(j.auroc_b0.mean()), "diff_vs_b0": t["mean"], "diff_b0_ci_low": t["ci_low"],
                                     "diff_b0_ci_high": t["ci_high"], "t_nb_b0": t["t"], "p_nb_b0": t["p"], "n_pairs_b0": t["n"]})
                rows.append(base)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Holm trong họ = (cohort, method, loại so sánh) qua các mức k
    for pcol in ("p_nb_random", "p_nb_b0"):
        if pcol in df.columns:
            df[f"{pcol}_holm"] = np.nan
            for _, idx in df.groupby(["cohort", "method"]).groups.items():
                df.loc[idx, f"{pcol}_holm"] = holm(df.loc[idx, pcol].to_numpy())
    if "p_nb_random_holm" in df.columns:
        df["beats_random_holm"] = df["p_nb_random_holm"] < C.ALPHA
        df["useful_rule_p0"] = df["beats_random_holm"] & (df["gain_vs_random"] >= 0.02)
    if "p_nb_b0_holm" in df.columns:
        df["differs_from_b0_holm"] = df["p_nb_b0_holm"] < C.ALPHA
    return df


def e1_bayes(per_fold: pd.DataFrame, cohort: str, classifier: str, rope: float = C.ROPE_AUROC, ks=C.ROPE_K, matrix_k: int = C.OVERLAP_K,
             full_matrix: bool = True):
    pf = per_fold[(per_fold.cohort == cohort) & (per_fold.classifier == classifier) & (~per_fold.method.map(is_reference))]
    pf = pf[pf.k.isin(list(ks) + [matrix_k])]
    if pf.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows, matrix_rows = [], []
    for k, g in pf.groupby("k"):
        if int(k) not in ks and int(k) != matrix_k:
            continue
        means = g.groupby("method")["auroc"].mean().sort_values(ascending=False)
        best = means.index[0]
        gb = g[g.method == best][["repeat", "fold", "auroc"]]
        for other in means.index[1:]:
            go = g[g.method == other][["repeat", "fold", "auroc"]]
            j = gb.merge(go, on=["repeat", "fold"], suffixes=("_best", "_other"))
            b = bayesian_correlated_ttest((j.auroc_best - j.auroc_other).to_numpy(), rope=rope)
            rows.append({"cohort": cohort, "k": int(k), "best": best, "other": other, "n_pairs": b["n"], "mean_diff": b["mean"],
                         "p_best_better": b["p_right"], "p_rope": b["p_rope"], "p_other_better": b["p_left"], "rope": rope})
        if full_matrix and int(k) == matrix_k:
            ms = list(means.index)
            for a in ms:
                ga = g[g.method == a][["repeat", "fold", "auroc"]]
                for b_ in ms:
                    gb_ = g[g.method == b_][["repeat", "fold", "auroc"]]
                    j = ga.merge(gb_, on=["repeat", "fold"], suffixes=("_a", "_b"))
                    r = bayesian_correlated_ttest((j.auroc_a - j.auroc_b).to_numpy(), rope=rope) if a != b_ else None
                    matrix_rows.append({"cohort": cohort, "k": int(k), "method_row": a, "method_col": b_,
                                        "p_row_better": np.nan if r is None else r["p_right"],
                                        "p_rope": np.nan if r is None else r["p_rope"], "n_pairs": len(j)})
    return pd.DataFrame(rows), pd.DataFrame(matrix_rows)


# ---------------------------------------------------------------------------------------------------
# E2–E3
# ---------------------------------------------------------------------------------------------------
def e3_exponent_table(exps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in exps.itertuples(index=False):
        th = TABLE5_EXPONENTS.get(r.method, (None, None, ""))
        rows.append({"method": r.method, "family": r.family, "alpha_n_measured": r.alpha_n, "alpha_n_table5": th[0],
                     "alpha_p_measured": r.alpha_p, "alpha_p_large_measured": getattr(r, "alpha_p_large", np.nan),
                     "alpha_p_table5": th[1], "complexity_table5": th[2], "n_range": r.n_range, "p_range": r.p_range,
                     "wall_spread_max_rel": getattr(r, "wall_spread_max_rel", np.nan)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------
# E6
# ---------------------------------------------------------------------------------------------------
def pareto_flags(df: pd.DataFrame, maximize=("auroc_mean", "nogueira"), minimize=("wall_s_e2",)) -> np.ndarray:
    vals = np.column_stack([df[c].to_numpy(float) for c in maximize] + [-df[c].to_numpy(float) for c in minimize])
    ok = ~np.isnan(vals).any(axis=1)
    flags = np.zeros(len(df), dtype=bool)
    idx = np.flatnonzero(ok)
    for i in idx:
        dominated = False
        for j in idx:
            if i != j and np.all(vals[j] >= vals[i]) and np.any(vals[j] > vals[i]):
                dominated = True
                break
        flags[i] = not dominated
    return flags


def e6_tradeoff(summary: pd.DataFrame, stab: pd.DataFrame | None, cost: pd.DataFrame | None, fraction: float, classifier: str,
                k_cost: int = C.OVERLAP_K):
    cols = [c for c in ("cohort", "method", "k", "n_folds", "auroc_mean", "auroc_sd", "auroc_ci_low", "auroc_ci_high", "select_time_s_mean")
            if c in summary.columns]
    s = summary[(summary.classifier == classifier) & (~summary.method.isin(["b0"]))][cols]
    out = s.copy()
    if stab is not None and len(stab):
        st = stab[np.isclose(stab.fraction, fraction)][["cohort", "method", "k", "nogueira", "nogueira_ci_low", "nogueira_ci_high",
                                                        "kuncheva", "jaccard_mean", "M"]]
        out = out.merge(st, on=["cohort", "method", "k"], how="left")
        out["stability_fraction"] = fraction
    if cost is not None and len(cost):
        c = cost[cost.k == k_cost] if "k" in cost.columns else cost
        mem_col = "maxrss_isolated_mb" if "maxrss_isolated_mb" in c.columns else None
        c = c[["method", "wall_s", "cpu_s"] + ([mem_col] if mem_col else [])].drop_duplicates("method").rename(
            columns={"wall_s": "wall_s_e2", "cpu_s": "cpu_s_e2", "maxrss_isolated_mb": "maxrss_isolated_mb_e2"})
        out = out.merge(c, on="method", how="left")
        out["cost_k"] = k_cost
    if "nogueira" in out.columns and "wall_s_e2" in out.columns:
        out["pareto_optimal"] = False
        for (co, k), idx in out.groupby(["cohort", "k"]).groups.items():
            sub = out.loc[idx]
            out.loc[idx, "pareto_optimal"] = pareto_flags(sub)
    return out


# ---------------------------------------------------------------------------------------------------
# E1b (đường cong học) và E7 (chuyển giao đơn giản)
# ---------------------------------------------------------------------------------------------------
def _n_t_key(v):
    return 10**9 if str(v) == "pool" else int(v)


def _boot_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def e1b_curves(per_run: pd.DataFrame, delta: float = C.KSTAR_DELTA) -> pd.DataFrame:
    """Trung bình ± SD, CI bootstrap qua các lần chạy ở mỗi (cohort, method, k, n_T); n_T* theo phương pháp và k."""
    rows = []
    for (co, m, k, n_t), g in per_run.groupby(["cohort", "method", "k", "n_t"]):
        x = g.auroc.to_numpy(float)
        lo, hi = _boot_ci(x, seed=abs(hash((co, m, int(k), str(n_t)))) % (2**31))
        rows.append({"cohort": co, "method": m, "k": int(k), "n_t": str(n_t), "n_runs": int(np.sum(~np.isnan(x))),
                     "auroc_mean": float(np.nanmean(x)) if np.any(~np.isnan(x)) else np.nan, "auroc_sd": float(np.nanstd(x, ddof=1)) if np.sum(~np.isnan(x)) > 1 else np.nan,
                     "auroc_ci_low": lo, "auroc_ci_high": hi,
                     "bacc_mean": float(g.bacc.mean()), "logloss_mean": float(g.logloss.mean()) if "logloss" in g else np.nan,
                     "brier_mean": float(g.brier.mean()) if "brier" in g else np.nan,
                     "n_filled_mean": float(pd.to_numeric(g.get("n_filled", pd.Series(dtype=float)), errors="coerce").mean()) if "n_filled" in g else np.nan})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # n_T*: mức n_T nhỏ nhất mà AUROC trung bình ≥ AUROC(pool) − Δ (mô tả; cùng tập con nên ghép cặp)
    out = []
    for (co, m, k), g in df.groupby(["cohort", "method", "k"]):
        g = g.sort_values("n_t", key=lambda c: c.map(_n_t_key))
        pool = g[g.n_t == "pool"]
        ref = float(pool.auroc_mean.iloc[0]) if len(pool) else np.nan
        n_t_star = None
        for r in g.itertuples():
            if r.n_t != "pool" and not np.isnan(ref) and r.auroc_mean >= ref - delta:
                n_t_star = int(r.n_t)
                break
        g = g.assign(auroc_pool=ref, n_t_star=n_t_star)
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    df["method_order"] = df["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
    return df.sort_values(["cohort", "k", "method_order", "n_t"], key=lambda c: c.map(_n_t_key) if c.name == "n_t" else c).drop(columns="method_order")


def e7_table(per_run: pd.DataFrame) -> pd.DataFrame:
    """Chênh lệch ghép cặp (chiến lược − target_only) theo (cohort, n_T, k): trung bình, CI bootstrap qua các lần chạy, p dấu (Wilcoxon), Holm trên n_T."""
    from scipy import stats as st
    key = ["cohort", "split", "subset", "n_t", "k"]
    base = per_run[per_run.strategy == "target_only"][key + ["auroc"]].rename(columns={"auroc": "auroc_target_only"})
    rows = []
    for strat in [x for x in per_run.strategy.unique() if x != "target_only"]:
        g = per_run[per_run.strategy == strat].merge(base, on=key, how="inner")
        for (co, n_t, k), h in g.groupby(["cohort", "n_t", "k"]):
            d = (h.auroc - h.auroc_target_only).to_numpy(float)
            lo, hi = _boot_ci(d, seed=abs(hash((co, strat, str(n_t), int(k)))) % (2**31))
            try:
                p = float(st.wilcoxon(d, alternative="two-sided").pvalue) if np.any(d != 0) else 1.0
            except ValueError:
                p = np.nan
            rows.append({"cohort": co, "strategy": strat, "n_t": str(n_t), "k": int(k), "n_runs": len(d),
                         "auroc_strategy": float(h.auroc.mean()), "auroc_target_only": float(h.auroc_target_only.mean()),
                         "diff_mean": float(d.mean()), "diff_ci_low": lo, "diff_ci_high": hi, "p_wilcoxon": p,
                         "negative_transfer": bool(hi < 0), "positive_transfer": bool(lo > 0)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_wilcoxon_holm"] = np.nan
    for _, idx in df.groupby(["cohort", "strategy", "k"]).groups.items():     # họ Holm = các mức n_T
        df.loc[idx, "p_wilcoxon_holm"] = holm(df.loc[idx, "p_wilcoxon"].to_numpy())
    return df.sort_values(["cohort", "strategy", "k", "n_t"], key=lambda c: c.map(_n_t_key) if c.name == "n_t" else c)


# ---------------------------------------------------------------------------------------------------
# Hình
# ---------------------------------------------------------------------------------------------------
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "grid.linewidth": 0.5, "lines.linewidth": 1.2, "lines.markersize": 4.5, "figure.dpi": 150,
                         "savefig.bbox": "tight"})
    return plt


def _savefig(fig, fig_out: Path, name: str, man: Manifest):
    for ext in ("pdf", "png"):
        p = fig_out / f"{name}.{ext}"
        fig.savefig(p)
        man.outputs.append(str(p))
    import matplotlib.pyplot as plt
    plt.close(fig)


def fig_e1(summary: pd.DataFrame, tests: pd.DataFrame, cohort: str, classifier: str, fig_out: Path, man: Manifest):
    plt = _style()
    s = summary[(summary.cohort == cohort) & (summary.classifier == classifier)]
    if s.empty:
        return
    fams = [("filter", C.FILTER_METHODS), ("embedded", C.EMBEDDED_METHODS), ("constrained", C.CONSTRAINED_METHODS), ("wrapper", C.WRAPPER_METHODS)]
    fams = [(f, [m for m in ms if m in set(s.method)]) for f, ms in fams]
    fams = [(f, ms) for f, ms in fams if ms]
    fig, axes = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.0), sharey=True, squeeze=False, constrained_layout=True)
    b0 = s[s.method == "b0"]
    rnd = tests[(tests.cohort == cohort)].groupby("k")["random_mean"].mean() if len(tests) and "random_mean" in tests else None
    ks = sorted(s[~s.method.map(is_reference)].k.unique())
    ihc = s[s.method.isin(["ihc_2", "ihc_4"])]
    for i, (ax, (fam, ms)) in enumerate(zip(axes[0], fams)):
        if len(b0):
            ax.axhline(b0.auroc_mean.iloc[0], color=FAMILY_COLOR["reference"], ls="-", lw=1, label="B0 (all genes)" if i == 0 else None)
        for r in ihc.itertuples():
            if r.k in ks:
                ax.scatter([ks.index(r.k)], [r.auroc_mean], marker=METHOD_MARKER.get(r.method, "*"), color=FAMILY_COLOR["knowledge"], s=40,
                           zorder=4, label=METHOD_LABEL.get(r.method) if i == 0 else None)
        if rnd is not None and len(rnd):
            r = rnd.reindex(ks)
            ax.plot(range(len(ks)), r.values, ls="--", color=FAMILY_COLOR["reference"], marker="+",
                    label="random-k (mean of draws)" if i == 0 else None)
        for m in ms:
            g = s[s.method == m].set_index("k").reindex(ks)
            x = np.arange(len(ks))
            ax.errorbar(x, g.auroc_mean, yerr=[g.auroc_mean - g.auroc_ci_low, g.auroc_ci_high - g.auroc_mean],
                        color=FAMILY_COLOR[fam], ls=METHOD_LS.get(m, "-"), marker=METHOD_MARKER.get(m, "o"), capsize=2, elinewidth=0.7,
                        label=METHOD_LABEL.get(m, m))
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks)
        ax.set_xlabel("k (genes)")
        ax.set_title(fam)
        ax.axhline(0.5, color="gray", lw=0.6, ls=":")
        ax.legend(loc="lower left", frameon=False, ncol=1)
    axes[0][0].set_ylabel(f"AUROC ({classifier}), mean ± 95% CI")
    fig.suptitle(f"E1 · {cohort}", fontsize=9)
    _savefig(fig, fig_out, f"fig_e1_auroc_vs_k_{cohort}", man)


def fig_e3(dn: pd.DataFrame, dp: pd.DataFrame, fig_out: Path, man: Manifest):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True)
    p_fixed = int(dn.p.iloc[0]) if len(dn) else C.PREFILTER_P0
    n_all = int(dp.n.iloc[0]) if len(dp) else None
    for ax, d, var, xl in ((axes[0], dn, "n", f"n (train samples), p = {p_fixed}"), (axes[1], dp, "p", f"p (genes), n = {n_all}")):
        for m, g in d.groupby("method"):
            g = g.sort_values(var)
            fam = C.METHOD_FAMILY.get(m, "reference")
            ax.plot(g[var], g.wall_s, color=FAMILY_COLOR[fam], ls=METHOD_LS.get(m, "-"), marker=METHOD_MARKER.get(m, "o"),
                    label=METHOD_LABEL.get(m, m))
            ax.annotate(METHOD_LABEL.get(m, m), (g[var].iloc[-1], g.wall_s.iloc[-1]), fontsize=6, color=FAMILY_COLOR[fam],
                        xytext=(3, 0), textcoords="offset points", va="center")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("wall time (s), median of 3")
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("E3 · scaling of the selection step (BLAS 1 thread)", fontsize=9)
    _savefig(fig, fig_out, "fig_e3_scaling", man)


def fig_e4(stab: pd.DataFrame, cohort: str, fig_out: Path, man: Manifest, k: int = C.OVERLAP_K):
    plt = _style()
    s = stab[(stab.cohort == cohort) & (stab.k == k)]
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(3.8, 3.0), constrained_layout=True)
    for m, g in s.groupby("method"):
        g = g.sort_values("fraction")
        fam = C.METHOD_FAMILY.get(m, "reference")
        ax.errorbar(g.fraction, g.nogueira, yerr=[g.nogueira - g.nogueira_ci_low, g.nogueira_ci_high - g.nogueira],
                    color=FAMILY_COLOR[fam], ls=METHOD_LS.get(m, "-"), marker=METHOD_MARKER.get(m, "o"), capsize=2,
                    elinewidth=0.7, label=METHOD_LABEL.get(m, m))
        ax.annotate(METHOD_LABEL.get(m, m), (g.fraction.iloc[-1], g.nogueira.iloc[-1]), fontsize=6, color=FAMILY_COLOR[fam],
                    xytext=(3, 0), textcoords="offset points", va="center")
    ax.set_xlabel("subsample fraction of patients")
    ax.set_ylabel(f"Nogueira stability (k = {k}), ± 95% CI")
    ax.set_title(f"E4 · {cohort}")
    ax.legend(frameon=False, ncol=2)
    _savefig(fig, fig_out, f"fig_e4_stability_{cohort}", man)


def fig_e5c(mat: pd.DataFrame, cohort: str, fig_out: Path, man: Manifest):
    plt = _style()
    names = [m for m in METHOD_ORDER if m in mat.index]
    M = mat.loc[names, names].to_numpy(float)
    fig, ax = plt.subplots(figsize=(3.9, 3.4), constrained_layout=True)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels([METHOD_LABEL.get(m, m) for m in names], rotation=60, ha="right")
    ax.set_yticklabels([METHOD_LABEL.get(m, m) for m in names])
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if M[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, label=f"mean Jaccard (k = {C.OVERLAP_K})", shrink=0.8)
    ax.set_title(f"E5c · {cohort}")
    _savefig(fig, fig_out, f"fig_e5c_overlap_{cohort}", man)


def fig_e6(trade: pd.DataFrame, cohort: str, fig_out: Path, man: Manifest):
    plt = _style()
    t = trade[(trade.cohort == cohort) & trade.nogueira.notna()]
    if t.empty:
        return
    ks = sorted(t.k.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(2.7 * len(ks), 3.1), sharey=True, sharex=True, squeeze=False,
                             constrained_layout=True)
    has_cost = "wall_s_e2" in t.columns and t.wall_s_e2.notna().any()
    for ax, k in zip(axes[0], ks):
        g = t[t.k == k]
        for r in g.itertuples():
            fam = C.METHOD_FAMILY.get(r.method, "reference")
            size = 12 + 10 * np.log10(max(r.wall_s_e2, 1e-2) + 1) * 4 if has_cost and not pd.isna(r.wall_s_e2) else 20
            ax.errorbar(r.nogueira, r.auroc_mean, xerr=[[r.nogueira - r.nogueira_ci_low], [r.nogueira_ci_high - r.nogueira]],
                        yerr=[[r.auroc_mean - r.auroc_ci_low], [r.auroc_ci_high - r.auroc_mean]], fmt="none",
                        ecolor=FAMILY_COLOR[fam], elinewidth=0.6, alpha=0.7)
            ax.scatter([r.nogueira], [r.auroc_mean], s=size, color=FAMILY_COLOR[fam], marker=METHOD_MARKER.get(r.method, "o"),
                       edgecolor="black" if getattr(r, "pareto_optimal", False) else "none", linewidth=0.8, zorder=3)
            ax.annotate(METHOD_LABEL.get(r.method, r.method), (r.nogueira, r.auroc_mean), fontsize=5.5, xytext=(3, 3),
                        textcoords="offset points", color=FAMILY_COLOR[fam])
        ax.set_title(f"k = {k}")
        ax.set_xlabel("Nogueira stability")
    axes[0][0].set_ylabel("AUROC (mean ± 95% CI)")
    note = f"marker size ∝ log wall time (E2, k = {C.OVERLAP_K}); black edge = Pareto-optimal" if has_cost else "black edge = Pareto-optimal"
    fig.suptitle(f"E6 · {cohort} · stability – performance – cost ({note})", fontsize=8)
    _savefig(fig, fig_out, f"fig_e6_tradeoff_{cohort}", man)


def fig_e1b(curves: pd.DataFrame, cohort: str, fig_out: Path, man: Manifest):
    plt = _style()
    c = curves[(curves.cohort == cohort) & (curves.n_t != "pool")]
    if c.empty:
        return
    ks = sorted(c.k.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(2.8 * len(ks), 3.0), sharey=True, squeeze=False, constrained_layout=True)
    for ax, k in zip(axes[0], ks):
        g = c[c.k == k]
        levels = sorted(g.n_t.unique(), key=_n_t_key)
        for m, h in g.groupby("method"):
            h = h.set_index("n_t").reindex(levels)
            fam = C.METHOD_FAMILY.get(m, "reference")
            x = np.arange(len(levels))
            ax.errorbar(x, h.auroc_mean, yerr=[h.auroc_mean - h.auroc_ci_low, h.auroc_ci_high - h.auroc_mean], color=FAMILY_COLOR[fam],
                        ls=METHOD_LS.get(m, "-"), marker=METHOD_MARKER.get(m, "o"), capsize=2, elinewidth=0.6, label=METHOD_LABEL.get(m, m))
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels)
        ax.set_xlabel("n_T per class")
        ax.set_title(f"k = {k}")
        ax.axhline(0.5, color="gray", lw=0.6, ls=":")
    axes[0][0].set_ylabel("AUROC on test_T (mean ± 95% bootstrap CI)")
    axes[0][-1].legend(frameon=False, ncol=1, fontsize=6, loc="lower right")
    fig.suptitle(f"E1b · {cohort} · learning curves of existing selectors (target-only)", fontsize=9)
    _savefig(fig, fig_out, f"fig_e1b_learning_curve_{cohort}", man)


def fig_e7(tab: pd.DataFrame, cohort: str, fig_out: Path, man: Manifest):
    plt = _style()
    t = tab[tab.cohort == cohort]
    if t.empty:
        return
    ks = sorted(t.k.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(2.8 * len(ks), 3.0), sharey=True, squeeze=False, constrained_layout=True)
    colors = {"source_only": "#2a78d6", "select_on_source": "#eb6834", "pooled": "#1baf7a"}
    for ax, k in zip(axes[0], ks):
        g = t[t.k == k]
        levels = sorted(g.n_t.unique(), key=_n_t_key)
        for strat, h in g.groupby("strategy"):
            h = h.set_index("n_t").reindex(levels)
            x = np.arange(len(levels))
            ax.errorbar(x, h.diff_mean, yerr=[h.diff_mean - h.diff_ci_low, h.diff_ci_high - h.diff_mean], color=colors.get(strat, "#6b6b6b"),
                        marker="o", capsize=2, elinewidth=0.6, label=strat)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels)
        ax.set_xlabel("n_T per class")
        ax.set_title(f"k = {k}")
    axes[0][0].set_ylabel("ΔAUROC vs target-only (paired, 95% bootstrap CI)")
    axes[0][-1].legend(frameon=False, fontsize=6)
    fig.suptitle(f"E7 · {cohort} · naive transfer strategies vs target-only", fontsize=9)
    _savefig(fig, fig_out, f"fig_e7_transfer_{cohort}", man)


# ---------------------------------------------------------------------------------------------------
def main(argv=None, script=__file__) -> None:
    args = parse_args(argv)
    prov = start_run(script, params=vars(args))
    root = DATA_ROOT.joinpath(*C.RESULT_DIR)
    out = results_dir(args.out)
    fig_out = results_dir(args.fig_out)
    man = Manifest()
    status = {}

    # ---------------- E0 ----------------
    e0_root = root / args.e0_dir
    e0_all = []
    if e0_root.exists():
        for d in sorted(p for p in e0_root.iterdir() if p.is_dir()):
            t = man.read_csv(d / "e0_signal_check.csv")
            if t is not None:
                e0_all.append(t)
    if e0_all:
        e0 = pd.concat(e0_all, ignore_index=True)
        save(e0, out, "e0_signal_check", man, tex=("E0: signal check per cohort (F-test $\\to$ LR-L2 at $k$ = 10; label permutations and random-$k$ panels).", "tab:e0"))
        status["E0"] = sorted(e0.cohort.unique())
    else:
        status["E0"] = "chưa có"

    # ---------------- E1 ----------------
    per_fold, random_k = load_phase_b(root / args.e1_dir, root / args.e1_select_dir, man)
    summary = tests = pd.DataFrame()
    if per_fold is not None:
        summary = e1_summary(per_fold)
        save(summary, out, "e1_summary", man)
        main_tab = summary[summary.classifier == args.classifier].copy()
        main_tab["AUROC"] = [fmt_pm(m, s) for m, s in zip(main_tab.auroc_mean, main_tab.auroc_sd)]
        wide = main_tab[main_tab.method != "b0"].pivot_table(index=["cohort", "method"], columns="k", values="AUROC",
                                                             aggfunc="first").reset_index()
        wide.columns = [str(c) for c in wide.columns]
        save(wide, out, "e1_auroc_table", man, tex=("E1: AUROC (mean $\\pm$ SD over repeated patient-level CV) by method and budget $k$; knowledge panels at their own $k$.", "tab:e1_auroc"))
        b0_tab = main_tab[main_tab.method == "b0"][[c for c in ("cohort", "k", "n_folds", "AUROC", "bacc_mean", "mcc_mean", "logloss_mean", "brier_mean") if c in main_tab.columns]].rename(columns={"k": "n_genes"})
        if len(b0_tab):
            save(b0_tab, out, "e1_b0_table", man, tex=("E1: reference B0 (all genes of the gene space, no selection) with the same classifier and CV protocol.", "tab:e1_b0"))
        tests = e1_tests(per_fold, random_k, args.classifier)
        if len(tests):
            save(tests, out, "e1_tests_vs_random_b0", man)
        # k* gộp (QĐ11) cho mọi phương pháp trên các cohort GPL570 có đủ dữ liệu
        pf_main = per_fold[per_fold.classifier == args.classifier]
        pooled_cohorts = [c for c in C.POOLED_KSTAR_COHORTS if c in set(pf_main.cohort)]
        methods_k = [m for m in pf_main.method.unique() if m not in ("b0", "random_k")]
        if pooled_cohorts:
            ks_tab, ks_sum = kstar_table(pf_main, methods_k, cohorts=pooled_cohorts, require_all_cohorts=True)
            if len(ks_tab):
                save(ks_tab, out, "e1_kstar_pooled_by_k", man)
                save(ks_sum, out, "e1_kstar_pooled", man, tex=(f"E1: minimum non-inferior panel size $k^*$ per method pooled over {', '.join(pooled_cohorts)} (QĐ11: IVW with Nadeau--Bengio variance and equal weights; fixed sequence).", "tab:e1_kstar"))
            pc = per_cohort_kstar(pf_main, methods_k, sorted(pf_main.cohort.unique()))
            save(pc, out, "e1_kstar_per_cohort", man)
        bayes_all, mat_all = [], []
        for co in sorted(pf_main.cohort.unique()):
            b, bm = e1_bayes(per_fold, co, args.classifier, full_matrix=(co == C.RANKING_DATASET))
            if len(b):
                bayes_all.append(b)
            if len(bm):
                mat_all.append(bm)
        if bayes_all:
            save(pd.concat(bayes_all, ignore_index=True), out, "e1_bayes_rope", man)
        if mat_all:
            save(pd.concat(mat_all, ignore_index=True), out, f"e1_bayes_rope_matrix_k{C.OVERLAP_K}", man)
        if random_k is not None:
            save(random_per_fold(random_k), out, "e1_random_k_per_fold", man)
        status["E1"] = {"cohorts": sorted(per_fold.cohort.unique()), "methods": sorted(per_fold.method.unique()),
                        "n_rows": int(len(per_fold)), "kstar_cohorts": pooled_cohorts}
        if not args.no_figures:
            for co in per_fold.cohort.unique():
                fig_e1(summary, tests, co, args.classifier, fig_out, man)
    else:
        status["E1"] = "chưa có"
        print("E1: chưa có kết quả (pha B)", flush=True)

    # ---------------- E1b ----------------
    e1b_root = root / args.e1b_dir
    e1b_all, e1b_stab = [], []
    if e1b_root.exists():
        for d in sorted(p for p in e1b_root.iterdir() if p.is_dir()):
            t = man.read_csv(d / "e1b_per_run.csv")
            if t is not None:
                e1b_all.append(t.assign(n_t=t.n_t.astype(str)))
            st_ = man.read_csv(d / "e1b_stability.csv")
            if st_ is not None:
                e1b_stab.append(st_)
    if e1b_all:
        e1b = pd.concat(e1b_all, ignore_index=True)
        curves = e1b_curves(e1b)
        save(curves, out, "e1b_learning_curves", man)
        wide = curves.assign(val=[fmt_pm(m, s) for m, s in zip(curves.auroc_mean, curves.auroc_sd)]).pivot_table(
            index=["cohort", "k", "method"], columns="n_t", values="val", aggfunc="first").reset_index()
        wide.columns = [str(c) for c in wide.columns]
        save(wide, out, "e1b_learning_curves_table", man, tex=("E1b: AUROC (mean $\\pm$ SD over 20 few-shot runs) of existing selectors by target sample size $n_T$ per class and budget $k$.", "tab:e1b"))
        if e1b_stab:
            save(pd.concat(e1b_stab, ignore_index=True), out, "e1b_stability", man)
        status["E1b"] = sorted(e1b.cohort.unique())
        if not args.no_figures:
            for co in e1b.cohort.unique():
                fig_e1b(curves, co, fig_out, man)
    else:
        status["E1b"] = "chưa có"

    # ---------------- E7 ----------------
    e7_root = root / args.e7_dir
    e7_all = []
    if e7_root.exists():
        for d in sorted(p for p in e7_root.iterdir() if p.is_dir()):
            t = man.read_csv(d / "e7_per_run.csv")
            if t is not None:
                e7_all.append(t.assign(n_t=t.n_t.astype(str)))
    if e7_all:
        e7 = pd.concat(e7_all, ignore_index=True)
        tab7 = e7_table(e7)
        if len(tab7):
            save(tab7, out, "e7_transfer_table", man, tex=("E7: paired difference in AUROC between each naive transfer strategy and target-only selection (same few-shot subsets), with 95\\% bootstrap CI and Holm-adjusted Wilcoxon $p$ over the $n_T$ levels.", "tab:e7"))
            if not args.no_figures:
                for co in tab7.cohort.unique():
                    fig_e7(tab7, co, fig_out, man)
        status["E7"] = sorted(e7.cohort.unique())
    else:
        status["E7"] = "chưa có"

    # ---------------- E2–E3 ----------------
    e23 = root / args.e2e3_dir
    cost = man.read_csv(e23 / "e2_cost_point.csv") if e23.exists() else None
    if cost is not None:
        c = cost.copy()
        c["wall_s"] = c["wall_s"].astype(float)
        # Y19/QĐ27: bộ nhớ đỉnh CHỈ từ tiến trình con (maxrss_isolated_mb, VmHWM); không dùng peak_mb (tracemalloc) hay maxrss_mb
        if "maxrss_isolated_mb" not in c.columns:
            print("E2: không có maxrss_isolated_mb (chạy với --isolate-memory) — bảng E2 không có cột bộ nhớ", flush=True)
        save(c[[x for x in ("session", "method", "family", "n", "p", "k", "wall_s", "cpu_s", "maxrss_isolated_mb", "peak_source", "repeats", "pop", "generations")
                if x in c.columns]], out, "e2_cost_table", man,
             tex=("E2: cost of the selection step at fixed points ($n$ = training part, $p$ = 5000, $k \\in \\{10, 50\\}$; median of 3 runs, "
                  "one Kaggle machine type with cross-session calibration, BLAS single-threaded; peak memory = max RSS of an isolated process).", "tab:e2_cost"))
        dn, dp = man.read_csv(e23 / "e3_scaling_n.csv"), man.read_csv(e23 / "e3_scaling_p.csv")
        exps = man.read_csv(e23 / "e3_exponents.csv")
        if exps is not None:
            save(e3_exponent_table(exps), out, "e3_exponents_vs_table5", man,
                 tex=("E3: empirical log--log exponents of wall time in $n$ and $p$ versus the theoretical orders of Table~\\ref{tab:fs_complexity}.", "tab:e3_exponents"))
        if dn is not None and dp is not None and not args.no_figures:
            fig_e3(dn, dp, fig_out, man)
        status["E2_E3"] = {"methods": sorted(cost.method.unique())}
    else:
        status["E2_E3"] = "chưa có"
        print("E2/E3: chưa có kết quả", flush=True)

    # ---------------- E4–E5 ----------------
    e45 = root / args.e4e5_dir
    stab_all, e5a_all, e5b_all = [], [], []
    overlaps = {}
    if e45.exists():
        for d in sorted(p for p in e45.iterdir() if p.is_dir()):
            s = man.read_csv(d / "e4_stability.csv")
            if s is not None:
                stab_all.append(s)
            a = man.read_csv(d / "e5a_redundancy.csv")
            if a is not None:
                e5a_all.append(a)
            b = man.read_csv(d / "e5b_seed_stability.csv")
            if b is not None:
                e5b_all.append(b)
            o = d / f"e5c_overlap_k{C.OVERLAP_K}.csv"
            if o.exists():
                man.read_csv(o)
                mat = pd.read_csv(o, index_col=0)
                if len(mat):
                    overlaps[d.name] = mat
    stab = pd.concat(stab_all, ignore_index=True) if stab_all else None
    if stab is not None:
        save(stab, out, "e4_stability", man)
        wide = stab.assign(val=[fmt_pm(m, (h - l) / 2) for m, l, h in zip(stab.nogueira, stab.nogueira_ci_low, stab.nogueira_ci_high)])
        wide = wide.pivot_table(index=["cohort", "fraction", "method"], columns="k", values="val", aggfunc="first").reset_index()
        wide.columns = [str(c) for c in wide.columns]
        save(wide, out, "e4_nogueira_table", man,
             tex=("E4: Nogueira stability ($\\pm$ half-width of the 95\\% CI) over $M$ = 30 independent patient subsamples, by subsample fraction and budget $k$.", "tab:e4_nogueira"))
        if not args.no_figures:
            for co in stab.cohort.unique():
                fig_e4(stab, co, fig_out, man)
        status["E4"] = {"cohorts": sorted(stab.cohort.unique())}
    else:
        status["E4"] = "chưa có"
    if e5a_all:
        e5a = pd.concat(e5a_all, ignore_index=True)
        save(e5a, out, "e5a_redundancy", man)
        cols = ["cohort", "rho", "method", "k", "nogueira", "nogueira_merged", "delta_nogueira_vs_e4", "pairs_both_selected", "n_copy_selected"]
        save(e5a[[c for c in cols if c in e5a.columns]], out, "e5a_redundancy_table", man,
             tex=("E5a: stability under injected redundancy (50 correlated copies, $\\rho \\in \\{0.5, 0.8, 0.95\\}$): raw and copy-merged Nogueira, change versus E4 at the same fraction, mean number of (source, copy) pairs selected together.", "tab:e5a_redundancy"))
        status["E5a"] = "có"
    if e5b_all:
        e5b = pd.concat(e5b_all, ignore_index=True)
        save(e5b, out, "e5b_seed_variability", man,
             tex=(f"E5b: seed sensitivity of stochastic selectors on one training fold of {C.RANKING_DATASET} (5 seeds): Nogueira/Jaccard between the five selected panels and the spread of test AUROC.", "tab:e5b_seed"))
        status["E5b"] = "có"
    for co, mat in overlaps.items():
        if not args.no_figures:
            fig_e5c(mat, co, fig_out, man)
        m2 = mat.copy()
        m2.insert(0, "method", m2.index)
        save(m2.reset_index(drop=True), out, f"e5c_overlap_k{C.OVERLAP_K}_{co}", man)
    if overlaps:
        status["E5c"] = sorted(overlaps)

    # ---------------- E6 ----------------
    if len(summary):
        trade = e6_tradeoff(summary, stab, cost, args.e6_fraction, args.classifier)
        save(trade, out, "e6_tradeoff", man)
        if "pareto_optimal" in trade.columns:
            save(trade[trade.pareto_optimal], out, "e6_pareto_front", man,
                 tex=("E6: Pareto-optimal (method, $k$) points in the AUROC -- stability -- cost space.", "tab:e6_pareto"))
        if not args.no_figures and "nogueira" in trade.columns:
            for co in trade.cohort.unique():
                fig_e6(trade, co, fig_out, man)
        status["E6"] = "có" if "nogueira" in trade.columns else "thiếu E4"

    man.save(out, prov)
    write_config(out, prov, {"experiment": "tables_figures", "status": status, "classifier": args.classifier,
                             "e6_fraction": args.e6_fraction, "n_inputs": len(man.inputs)})
    print(json.dumps(status, ensure_ascii=False, indent=1))
    print("xong:", out, "|", fig_out, flush=True)


if __name__ == "__main__":
    main()
