"""Thống kê RQ2 (Review 08/11/2026 Mục 8.4, QĐ69): H2, H2b, H2c từ điểm từng mẫu của JOB rq2_mvp; không chạy lại mô hình.

Phân tích mù (QĐ69b): script này và test của nó được commit TRƯỚC khi tính bất kỳ trung bình AUROC nào của RQ2 trên GEO.
config.json ghi thời điểm mở kết quả (results_opened_at_utc) để chép vào HANDOVER.

Chênh của một lần chạy (split, subset) = trung bình theo k của AUROC(MVP) − AUROC(đối chứng) trên cùng test_T. MVP là cấu hình
m5_pos (QĐ49): điểm lấy từ dòng gamma_rule = "fixed" của đúng γ và không gian gen mà dòng m5_pos ghi. Target-only = γ = 0 lọc trên L_T
(prefilter_space = "target"); nhánh cơ chế γ = 0/source không vào so sánh nào.

Bootstrap hai tầng, 2.000 vòng, cho mỗi (cohort, n_T, so sánh): tầng 1 rút có hoàn lại các lần chạy; tầng 2 rút có hoàn lại bệnh nhân
của test_T, phân tầng theo lớp, MỘT lần cho mỗi split trong mỗi vòng và dùng chung cho mọi tập con, mọi k, cả hai nhánh của split đó
(QĐ69a). AUROC trên mẫu đã rút là Mann–Whitney có trọng số (số lần mỗi bệnh nhân được rút). SE = độ lệch chuẩn của trung bình qua
các vòng; seed = CRC32 của khoá (QĐ30).

Gộp qua cohort: IVW (w = 1/SE²) và trọng số bằng nhau; chính là ba cohort GPL570, độ nhạy là 4 cohort. p một phía xấp xỉ chuẩn.
H2: Δ(MVP − target-only), k ∈ {2, 3, 5}, Holm trên n_T ∈ {5, 10, 20}; ủng hộ khi p Holm < 0,05 ở ≥ 1 mức và bản bằng nhau cùng dấu;
bác bỏ khi Δ IVW ≤ 0 ở cả ba mức. H2b: MVP so với IHC cùng k (k ∈ {2, 4}), n_T ∈ {5, 10}, cận dưới > −0,02, chỉ báo cáo.
H2c: MVP so với select-on-source, pooled, H0: Δ ≤ −0,01, Holm trên 6 kiểm định (KBTN v2.4 Mục 9).

    python scripts/make_rq2_tables.py --input rq2_mvp_1b7c021_results.zip
    python scripts/make_rq2_tables.py                      # đọc $KLTN_DATA_ROOT/results/survey_v2/RQ2_mvp

Đầu ra: results/survey_v2/RQ2_tables/{rq2_cells.csv, rq2_pooled.csv, rq2_h2b.csv, rq2_decisions.csv, config.json}.
Với --r-input (zip của notebook kaggle_rq2_r_run, STAGE = all): thêm so sánh MVP với ptLasso, Trans-GLM (QĐ68) theo cùng bootstrap hai tầng,
ghép cặp trên cùng test_T. KBTN không đăng ký giả thuyết cho hai so sánh này nên chỉ MÔ TẢ (Δ, SE, CI 95 % hai phía, gộp IVW và trọng số
bằng nhau; không Holm); thêm rq2_baselines.csv, rq2_ptlasso_degeneracy.csv (tỉ lệ same_as_stage1_topk theo cohort, n_T, k) và
rq2_tuning_budget.csv (PD-05). Các bảng H2, H2b, H2c không đổi khi thêm --r-input.
Cổng bằng nhau (rq2_equality_report.csv) thiếu hoặc có dòng không ok thì dừng, không tính gì (QĐ70).
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import sha256  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402
from survey_v2.stats import holm  # noqa: E402

MVP_RULE = "m5_pos"
K_AVG = (2, 3, 5)                                  # H2, H2c: trung bình theo k
H2_N_T = ("5", "10", "20")                         # họ Holm của H2
H2B = {"ihc_2": (2,), "ihc_4": (4,)}               # H2b: MVP ở đúng k của panel IHC
H2B_N_T = ("5", "10")
H2C_ARMS = ("select_on_source", "pooled")
D3_N_T = tuple(str(v) for v in C.TRANSFER_N_T)
MARGIN_H2B = 0.02
MARGIN_H2C = 0.01
ALPHA = 0.05
Z95 = float(norm.ppf(0.95))
AUROC_TOL = 1e-9                                   # AUROC tính lại từ điểm phải trùng per_run
POOL_SETS = {"gpl570": tuple(C.GPL570_COHORTS)}    # "all4" = mọi cohort đã nạp (độ nhạy)
# Baseline R (QĐ68): KBTN không đăng ký giả thuyết cho MVP so với ptLasso, Trans-GLM -> chỉ mô tả: Δ, SE, CI 95 % hai phía, gộp IVW và
# trọng số bằng nhau; không Holm, không kết luận. Cùng bootstrap hai tầng và ghép cặp trên cùng test_T như H2.
R_ARMS = ("ptlasso", "transglm")
R_FAMILY = "mô tả (không đăng ký giả thuyết; không Holm)"
Z975 = float(norm.ppf(0.975))
RUN = ["split", "subset"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=None, help="zip của JOB rq2_mvp hoặc thư mục RQ2_mvp (mặc định $KLTN_DATA_ROOT/results/survey_v2/RQ2_mvp)")
    ap.add_argument("--cohorts", nargs="+", default=list(C.FEWSHOT_COHORTS))
    ap.add_argument("--n-boot", type=int, default=C.BOOTSTRAP_N)
    ap.add_argument("--r-input", type=Path, default=None, help="zip của notebook kaggle_rq2_r_run (STAGE = all) hoặc thư mục RQ2_r: thêm so sánh mô tả với ptLasso, Trans-GLM")
    ap.add_argument("--out", default="RQ2_tables")
    return ap.parse_args(argv)


def load_r_inputs(src: Path, cohorts) -> dict:
    """{cohort: (rq2_r_per_run, rq2_r_scores, rq2_r_diagnostics)} từ zip của kaggle_rq2_r_run hoặc thư mục RQ2_r."""
    src = Path(src)
    names = ("rq2_r_per_run.csv", "rq2_r_scores.csv", "rq2_r_diagnostics.csv")
    out = {}
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as zf:
            members = zf.namelist()
            for c in cohorts:
                frames = []
                for name in names:
                    hits = [n for n in members if n.endswith(f"RQ2_r/{c}/{name}")]
                    if len(hits) != 1:
                        raise ValueError(f"{src}: cần đúng một RQ2_r/{c}/{name}, thấy {len(hits)}")
                    frames.append(_read(io.BytesIO(zf.read(hits[0]))))
                out[c] = tuple(frames)
        return out
    for c in cohorts:
        out[c] = tuple(_read(src / c / name) for name in names)
    return out


def degeneracy_table(diags: dict) -> pd.DataFrame:
    """Tỉ lệ same_as_stage1_topk của ptLasso theo (cohort, n_T, k) (QĐ68: báo trong chú thích bảng RQ2)."""
    rows = []
    for cohort, d in diags.items():
        pt = d[d.method == "ptlasso"].copy()
        pt["n_t"] = pt.n_t.astype(str)
        for (n_t, k), g in pt.groupby(["n_t", "k"]):
            v = pd.to_numeric(g.same_as_stage1_topk, errors="coerce")
            rows.append({"cohort": cohort, "n_t": n_t, "k": int(k), "n_runs": int(len(g)), "n_defined": int(v.notna().sum()),
                         "same_as_stage1_topk_rate": float(v.mean()) if v.notna().any() else np.nan,
                         "alpha_mode": float(g.alpha.mode().iloc[0]) if "alpha" in g and g.alpha.notna().any() else np.nan})
    t = pd.DataFrame(rows)
    return t.sort_values(["cohort", "n_t", "k"], key=lambda s: s.map(_n_t_key) if s.name == "n_t" else s).reset_index(drop=True) if len(t) else t


def tuning_budget() -> pd.DataFrame:
    """Bảng ngân sách tuning (PD-05, QĐ68): phần nào được chỉnh trên L_T. Bước phân loại giống nhau ở mọi nhánh (evaluate_panel, M5)."""
    clf = "LR-L2 (evaluate_panel, M5): n_T ≤ 10 C = 1, ngưỡng 0,5; n_T ≥ 20 C ∈ {0,01; 0,1; 1; 10} bằng CV 3 fold trong L_T"
    rows = [
        ("mvp_m5_pos", True, "γ ∈ {0,5; 1; 2} (3 ứng viên) chọn bằng log-loss OOF trong L_T (CV 5 fold; lặp 3 lần khi n_T ≥ 20), biên 0,01; L1 đúng k bằng dò C; "
                             "π_j từ stability selection trên nguồn (một lần)"),
        ("target_only", False, "Không có tham số chỉnh: L1 đúng k bằng dò C trên L_T (lọc phương sai trên L_T)"),
        ("select_on_source", False, "Không chỉnh trên L_T: panel chọn trên nguồn (L1 đúng k)"),
        ("pooled", False, "Không chỉnh trên L_T: L1 đúng k trên nguồn + L_T gộp"),
        ("ihc", False, "Panel cố định (IHC-2, IHC-4)"),
        ("ptlasso", True, "α ∈ {0,75; 0,5; 0,25} (3 ứng viên) bằng deviance CV trong L_T (fold của MVP), biên 0,01; λ2 = lambda.min trên đường λ tự sinh của "
                          "cv.glmnet; λ1 = lambda.1se cố định trên nguồn (CV 10 fold, một lần)"),
        ("transglm", True, "Theo mặc định của glmtrans: CV nội bộ chọn λ, phát hiện nguồn có hại (transfer.source.id = auto); không chỉnh thêm"),
    ]
    return pd.DataFrame([{"method": m_, "tuned_on_L_T": t_, "selection_stage": s_, "classifier_stage": clf} for m_, t_, s_ in rows])


# ------------------------------------------------------------------------------------------------- đọc dữ liệu
def _read(buf) -> pd.DataFrame:
    return pd.read_csv(buf, dtype={"n_t": str, "sample_id": str}, float_precision="round_trip")


def load_inputs(src: Path, cohorts) -> tuple[dict, pd.DataFrame | None]:
    """{cohort: (per_run, scores)} và rq2_equality_report.csv (None nếu không có) từ zip của JOB hoặc thư mục RQ2_mvp."""
    src = Path(src)
    data = {}
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as zf:
            names = zf.namelist()

            def member(rel):
                hits = [n for n in names if n.endswith("RQ2_mvp/" + rel)]
                if len(hits) > 1:
                    raise ValueError(f"{src}: nhiều hơn một RQ2_mvp/{rel}")
                return _read(io.BytesIO(zf.read(hits[0]))) if hits else None

            report = member("rq2_equality_report.csv")
            for c in cohorts:
                per, sc = member(f"{c}/rq2_mvp_per_run.csv"), member(f"{c}/rq2_mvp_scores.csv")
                if per is None or sc is None:
                    raise ValueError(f"{src}: thiếu per_run hoặc scores của {c}")
                data[c] = (per, sc)
        return data, report
    report_path = src / "rq2_equality_report.csv"
    report = _read(report_path) if report_path.exists() else None
    for c in cohorts:
        data[c] = (_read(src / c / "rq2_mvp_per_run.csv"), _read(src / c / "rq2_mvp_scores.csv"))
    return data, report


def check_gate(report) -> int:
    """QĐ70: cổng bằng nhau phải có và toàn ok trước mọi thống kê. Trả số dòng của báo cáo."""
    if isinstance(report, (str, Path)):
        path = Path(report)
        if not path.exists():
            raise SystemExit(f"KHÔNG tính: thiếu báo cáo cổng bằng nhau {path}")
        report = pd.read_csv(path)
    if report is None or len(report) == 0:
        raise SystemExit("KHÔNG tính: thiếu báo cáo cổng bằng nhau (rq2_equality_report.csv)")
    bad = report[report.status != "ok"]
    if len(bad):
        raise SystemExit(f"KHÔNG tính: cổng bằng nhau không đạt ở {len(bad)}/{len(report)} dòng (QĐ70: dừng, giảng viên quyết định)")
    return int(len(report))


def _normalize(per: pd.DataFrame, sc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pilot 9a15860 (trước QĐ51) chưa có cột method / prefilter_space trong scores và prefilter_space trong per_run."""
    per, sc = per.copy(), sc.copy()
    for df in (per, sc):
        df["n_t"] = df["n_t"].astype(str)
        if "method" not in df:
            df["method"] = "mvp"
        if "prefilter_space" not in df:
            df["prefilter_space"] = np.where(df.method != "mvp", "none", np.where(df.gamma.fillna(-1) == 0, "target", "source"))
        df["gamma_key"] = df["gamma"].fillna(-1.0).astype(float)
    return per, sc


# ------------------------------------------------------------------------------------------------- AUROC có trọng số
def pair_matrix(score: np.ndarray, y: np.ndarray) -> np.ndarray:
    """M[i, j] = 1 nếu điểm dương thứ i > điểm âm thứ j, 0,5 nếu bằng: AUROC = trung bình M; với số lần rút c⁺, c⁻ là c⁺ᵀ M c⁻ / (Σc⁺ Σc⁻)."""
    s, y = np.asarray(score, float), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    return (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])


@dataclass
class Run:
    split: int
    subset: int
    D: np.ndarray            # trung bình theo k của (M_a − M_b): n_pos × n_neg
    auroc_a: dict            # k -> AUROC của nhánh a (per_run)
    auroc_b: dict


def _arm_rows(per: pd.DataFrame, arm: str, n_t: str, ks, mvp_rule: str) -> pd.DataFrame:
    """arm: "mvp" (dòng quy tắc mvp_rule), "target_only", tên method (D3, IHC, baseline R), hoặc khai báo đầy đủ
    "method|gamma_rule|gamma|prefilter_space" (ô trống = không lọc), vd. "mvp|fixed|0|source" (γ = 0 trên không gian gen của nguồn),
    "mvp|m5_cv||", "mvp_noihc|m5_pos||" (nhánh của tệp khác đã đổi tên method)."""
    sel = (per.n_t == n_t) & per.k.isin(ks)
    if "|" in arm:
        method, rule, gamma, space = (arm.split("|") + ["", "", "", ""])[:4]
        if method:
            sel &= per.method == method
        if rule:
            sel &= per.gamma_rule == rule
        if gamma != "":
            sel &= per.gamma_key == float(gamma)
        if space:
            sel &= per.prefilter_space == space
    elif arm == "mvp":
        sel &= (per.gamma_rule == mvp_rule) & (per.method == "mvp")
    elif arm == "target_only":
        sel &= (per.gamma_rule == "fixed") & (per.method == "mvp") & (per.gamma_key == 0) & (per.prefilter_space == "target")
    else:
        sel &= per.method == arm
    rows = per[sel]
    if rows.empty:
        raise ValueError(f"không có dòng nào của {arm} ở n_T = {n_t}, k ∈ {list(ks)}")
    dup = rows.duplicated(RUN + ["k"])
    if dup.any():
        raise ValueError(f"{arm}, n_T = {n_t}: trùng dòng theo (split, subset, k)")
    return rows


def collect_runs(per: pd.DataFrame, sc: pd.DataFrame, cohort: str, n_t: str, arm_a: str, arm_b: str, ks, mvp_rule: str = MVP_RULE) -> list:
    """Các lần chạy của một ô: ghép điểm từng mẫu của hai nhánh, kiểm AUROC tính lại = per_run, kiểm test_T dùng chung trong split."""
    per, sc = _normalize(per, sc)
    ks = tuple(int(k) for k in ks)
    rows = {arm: _arm_rows(per, arm, n_t, ks, mvp_rule) for arm in (arm_a, arm_b)}
    keys = {arm: set(map(tuple, r[RUN + ["k"]].to_numpy().tolist())) for arm, r in rows.items()}
    if keys[arm_a] != keys[arm_b]:
        raise ValueError(f"{cohort} n_T = {n_t}: {arm_a} và {arm_b} không cùng tập (split, subset, k)")
    runs_k = {}
    for split, subset, k in keys[arm_a]:
        runs_k.setdefault((int(split), int(subset)), set()).add(int(k))
    if any(v != set(ks) for v in runs_k.values()):
        raise ValueError(f"{cohort} n_T = {n_t}: có lần chạy thiếu k trong {list(ks)}")
    sc = sc[sc.n_t == n_t]
    groups = {key: g for key, g in sc.groupby(["method", "gamma_key", "prefilter_space", "split", "subset", "k"])}
    test_by_split: dict = {}
    out = []
    for split, subset in sorted(runs_k):
        D, aucs = None, {arm_a: {}, arm_b: {}}
        for k in ks:
            mats = []
            for arm in (arm_a, arm_b):
                r = rows[arm]
                row = r[(r.split == split) & (r.subset == subset) & (r.k == k)].iloc[0]
                g = groups.get((row.method, float(row.gamma_key), row.prefilter_space, split, subset, k))
                if g is None or g.empty:
                    raise ValueError(f"{cohort} n_T = {n_t}: thiếu điểm của {arm} ở (split {split}, subset {subset}, k {k})")
                g = g.sort_values("sample_id")
                ids, y = tuple(g.sample_id), g.y_true.to_numpy(int)
                prev = test_by_split.setdefault(split, (ids, tuple(y)))
                if prev != (ids, tuple(y)):
                    raise ValueError(f"{cohort} n_T = {n_t}: split {split} có test_T khác nhau giữa các lần chạy / nhánh (tầng 2 cần test_T dùng chung)")
                M = pair_matrix(g.score.to_numpy(float), y)
                if abs(M.mean() - float(row.auroc)) > AUROC_TOL:
                    raise ValueError(f"{cohort} n_T = {n_t}: AUROC tính lại từ điểm của {arm} ({M.mean():.12f}) khác per_run ({row.auroc:.12f}) "
                                     f"ở (split {split}, subset {subset}, k {k})")
                aucs[arm][k] = float(row.auroc)
                mats.append(M)
            D = (mats[0] - mats[1]) if D is None else D + (mats[0] - mats[1])
        out.append(Run(split, subset, D / len(ks), aucs[arm_a], aucs[arm_b]))
    return out


# ------------------------------------------------------------------------------------------------- bootstrap hai tầng
def cell_seed(*parts) -> int:
    """Seed tái lập từ khoá (QĐ30): CRC32 của chuỗi khoá."""
    return zlib.crc32("|".join(str(p) for p in ("rq2", *parts)).encode("utf-8")) & 0x7FFFFFFF


def bootstrap_cell(runs: list, n_boot: int, seed: int, return_draws: bool = False):
    """Δ = trung bình các lần chạy của trung bình D; SE từ bootstrap hai tầng (tầng 2 một lần cho mỗi split mỗi vòng)."""
    if not runs:
        raise ValueError("ô không có lần chạy nào")
    rng = np.random.default_rng(seed)
    R = len(runs)
    tier1 = rng.integers(0, R, size=(n_boot, R))
    splits = sorted({r.split for r in runs})
    counts = {}
    for s in splits:
        n_pos, n_neg = next(r.D.shape for r in runs if r.split == s)
        if any(r.D.shape != (n_pos, n_neg) for r in runs if r.split == s):
            raise ValueError(f"split {s}: các lần chạy có test_T khác cỡ")
        counts[s] = (rng.multinomial(n_pos, np.full(n_pos, 1 / n_pos), size=n_boot),
                     rng.multinomial(n_neg, np.full(n_neg, 1 / n_neg), size=n_boot))
    vals = np.empty((n_boot, R))
    for j, r in enumerate(runs):
        cpos, cneg = counts[r.split]
        vals[:, j] = np.einsum("bi,ij,bj->b", cpos, r.D, cneg) / (r.D.shape[0] * r.D.shape[1])
    means = np.take_along_axis(vals, tier1, axis=1).mean(axis=1)
    delta = float(np.mean([r.D.mean() for r in runs]))
    se = float(np.std(means, ddof=1))
    cell = {"delta": delta, "se": se, "lower": delta - Z95 * se, "n_runs": R, "n_splits": len(splits), "n_boot": int(n_boot), "seed": int(seed)}
    return (cell, vals, counts) if return_draws else cell


# ------------------------------------------------------------------------------------------------- gộp, quy tắc
def pool(cells: pd.DataFrame, margin: float) -> dict:
    """IVW và trọng số bằng nhau qua cohort; p một phía của H0: Δ ≤ −margin; I² theo Cochran Q."""
    d, se = cells.delta.to_numpy(float), cells.se.to_numpy(float)
    d_eq, se_eq = float(d.mean()), float(np.sqrt((se**2).sum()) / len(d))
    if (se > 0).all():
        w = 1.0 / se**2
        d_ivw, se_ivw = float((w * d).sum() / w.sum()), float(1.0 / np.sqrt(w.sum()))
        q = float((w * (d - d_ivw) ** 2).sum())
    else:  # SE = 0 ở một cohort (hai nhánh trùng hệt ở mọi lần chạy, vd. ablation γ = γ của m5_pos): IVW không xác định
        w, d_ivw, se_ivw, q = np.full(len(d), np.nan), np.nan, np.nan, np.nan
    p_of = lambda x, s: float(norm.sf((x + margin) / s)) if s > 0 else np.nan  # noqa: E731
    out = {"n_cohorts": int(len(d)), "delta_ivw": d_ivw, "se_ivw": se_ivw, "lower_ivw": d_ivw - Z95 * se_ivw,
           "p_ivw": p_of(d_ivw, se_ivw), "delta_eq": d_eq, "se_eq": se_eq, "lower_eq": d_eq - Z95 * se_eq,
           "p_eq": p_of(d_eq, se_eq), "q": q, "i2": max(0.0, (q - (len(d) - 1)) / q) if q > 0 else (np.nan if np.isnan(q) else 0.0), "margin": margin,
           "ci95_ivw_low": d_ivw - Z975 * se_ivw, "ci95_ivw_high": d_ivw + Z975 * se_ivw, "ci95_eq_low": d_eq - Z975 * se_eq, "ci95_eq_high": d_eq + Z975 * se_eq}
    for c, wi in zip(cells.cohort, w / np.nansum(w) if np.isfinite(w).all() else w):
        out[f"w_{c}"] = float(wi)
    return out


def decide_h2(table: pd.DataFrame) -> tuple[str, list]:
    """Bảng 3 dòng (n_T 5, 10, 20) của bản gộp chính. Trả (kết luận, p Holm theo thứ tự dòng)."""
    p_holm = holm(table.p_ivw.to_numpy(float))
    sig = (p_holm < ALPHA) & (table.delta_eq.to_numpy(float) > 0)
    if sig.any():
        return "ung_ho", p_holm.tolist()
    if (table.delta_ivw.to_numpy(float) <= 0).all():
        return "bac_bo", p_holm.tolist()
    return "khong_ket_luan", p_holm.tolist()


def decide_h2c(table: pd.DataFrame) -> tuple[str, list]:
    """Bảng 6 dòng (2 chiến lược × 3 mức n_T). Ủng hộ: bác bỏ H0 ở cả 6 và mọi cận dưới bản bằng nhau > −0,01."""
    p_holm = holm(table.p_ivw.to_numpy(float))
    rej = p_holm < ALPHA
    if rej.all() and (table.lower_eq.to_numpy(float) > -MARGIN_H2C).all():
        return "ung_ho", p_holm.tolist()
    if rej.any():
        return "ung_ho_mot_phan", p_holm.tolist()
    return "khong_ung_ho", p_holm.tolist()


def _n_t_key(v: str) -> int:
    return 10**9 if v == "pool" else int(v)


def build(data: dict, n_boot: int, mvp_rule: str = MVP_RULE, r_data: dict | None = None) -> dict:
    """data = {cohort: (per_run, scores)}; r_data = {cohort: (per_run, scores, diagnostics)} của baseline R (tuỳ chọn).
    Trả {"cells", "pooled", "h2b", "decisions"} và, khi có r_data, {"baselines", "degeneracy", "tuning_budget"}."""
    specs = []                                                     # (comparison, ks, margin, mức n_T được phép)
    specs.append(("target_only", K_AVG, 0.0, None))
    specs += [(arm, K_AVG, MARGIN_H2C, D3_N_T) for arm in H2C_ARMS]
    specs += [(arm, ks, MARGIN_H2B, H2B_N_T) for arm, ks in H2B.items()]
    if r_data:
        specs += [(arm, K_AVG, 0.0, None) for arm in R_ARMS]
    cells = []
    for cohort, (per, sc) in data.items():
        if r_data and cohort in r_data:
            rp, rs, _ = r_data[cohort]
            per, sc = pd.concat([per, rp], ignore_index=True), pd.concat([sc, rs], ignore_index=True)
        levels = sorted(per.n_t.astype(str).unique(), key=_n_t_key)
        for comp, ks, margin, allowed in specs:
            if comp in R_ARMS and not (r_data and cohort in r_data):
                continue
            for n_t in levels if allowed is None else [v for v in levels if v in allowed]:
                runs = collect_runs(per, sc, cohort, n_t, "mvp", comp, ks, mvp_rule)
                seed = cell_seed(cohort, n_t, comp, *ks)
                cell = bootstrap_cell(runs, n_boot, seed)
                cells.append({"cohort": cohort, "comparison": comp, "n_t": n_t, "k_avg": " ".join(map(str, ks)), **cell,
                              "auroc_mvp_mean": float(np.mean([np.mean(list(r.auroc_a.values())) for r in runs])),
                              "auroc_ref_mean": float(np.mean([np.mean(list(r.auroc_b.values())) for r in runs]))})
    cells = pd.DataFrame(cells)
    margins = {comp: margin for comp, _, margin, _ in specs}
    pool_sets = {**POOL_SETS, "all4": tuple(data)}
    pooled = []
    for (comp, n_t), g in cells.groupby(["comparison", "n_t"], sort=False):
        for name, members in pool_sets.items():
            sub = g[g.cohort.isin(members)]
            if len(sub):
                pooled.append({"comparison": comp, "n_t": n_t, "pool_set": name, "cohorts": " ".join(sub.cohort), **pool(sub, margins[comp])})
    pooled = pd.DataFrame(pooled)
    pooled["p_holm"] = np.nan
    main = pooled.pool_set == "gpl570"
    decisions = []
    h2_idx = pooled.index[main & (pooled.comparison == "target_only") & pooled.n_t.isin(H2_N_T)]
    h2 = pooled.loc[h2_idx].sort_values("n_t", key=lambda s: s.map(_n_t_key))
    if len(h2) == len(H2_N_T):
        outcome, p_holm = decide_h2(h2)
        pooled.loc[h2.index, "p_holm"] = p_holm
        decisions.append({"hypothesis": "H2", "outcome": outcome, "family": "Holm, n_T ∈ {5, 10, 20}, bản gộp GPL570 (IVW)",
                          "detail": "; ".join(f"n_T={r.n_t}: Δ_ivw={r.delta_ivw:.4f}, p_holm={p:.4g}, Δ_eq={r.delta_eq:.4f}" for r, p in zip(h2.itertuples(), p_holm))})
    h2c_idx = pooled.index[main & pooled.comparison.isin(H2C_ARMS) & pooled.n_t.isin(D3_N_T)]
    h2c = pooled.loc[h2c_idx].sort_values(["comparison", "n_t"], key=lambda s: s.map(_n_t_key) if s.name == "n_t" else s.map(H2C_ARMS.index))
    if len(h2c) == len(H2C_ARMS) * len(D3_N_T):
        outcome, p_holm = decide_h2c(h2c)
        pooled.loc[h2c.index, "p_holm"] = p_holm
        decisions.append({"hypothesis": "H2c", "outcome": outcome, "family": "Holm, 6 kiểm định (2 chiến lược × 3 mức n_T), H0: Δ ≤ −0,01",
                          "detail": "; ".join(f"{r.comparison} n_T={r.n_t}: Δ_ivw={r.delta_ivw:.4f}, p_holm={p:.4g}, lower_eq={r.lower_eq:.4f}"
                                              for r, p in zip(h2c.itertuples(), p_holm))})
    h2b = pooled[pooled.comparison.isin(H2B)].copy()
    h2b["noninferior_both"] = (h2b.lower_ivw > -MARGIN_H2B) & (h2b.lower_eq > -MARGIN_H2B)
    out = {"cells": cells, "pooled": pooled, "h2b": h2b, "decisions": pd.DataFrame(decisions)}
    if r_data:
        base = pooled[pooled.comparison.isin(R_ARMS)].copy()
        base.insert(3, "family", R_FAMILY)
        out.update({"baselines": base.reset_index(drop=True), "degeneracy": degeneracy_table({c: d[2] for c, d in r_data.items()}),
                    "tuning_budget": tuning_budget()})
    return out


# ------------------------------------------------------------------------------------------------- CLI
def main(argv=None, script=__file__) -> int:
    args = parse_args(argv)
    src = args.input or DATA_ROOT.joinpath(*C.RESULT_DIR, "RQ2_mvp")
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t0 = time.time()
    data, report = load_inputs(src, args.cohorts)
    n_gate = check_gate(report)
    opened = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"cổng bằng nhau: {n_gate} dòng ok; mở kết quả lúc {opened}", flush=True)
    r_data = load_r_inputs(args.r_input, args.cohorts) if args.r_input else None
    tables = build(data, args.n_boot, r_data=r_data)
    out = results_dir(args.out)
    names = [("rq2_cells", "cells"), ("rq2_pooled", "pooled"), ("rq2_h2b", "h2b"), ("rq2_decisions", "decisions")]
    if r_data:
        names += [("rq2_baselines", "baselines"), ("rq2_ptlasso_degeneracy", "degeneracy"), ("rq2_tuning_budget", "tuning_budget")]
    for name, key in names:
        tables[key].to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    cfg = {"experiment": "RQ2_tables", "input": str(src), "input_sha256": sha256(src) if Path(src).suffix == ".zip" else None,
           "cohorts": args.cohorts, "pool_sets": {**POOL_SETS, "all4": list(args.cohorts)}, "mvp_rule": MVP_RULE, "k_avg": list(K_AVG),
           "h2_family_n_t": list(H2_N_T), "h2b": {k: list(v) for k, v in H2B.items()}, "h2b_n_t": list(H2B_N_T), "h2c_arms": list(H2C_ARMS),
           "margin_h2b": MARGIN_H2B, "margin_h2c": MARGIN_H2C, "alpha": ALPHA, "n_boot": args.n_boot,
           "bootstrap": "tầng 1: lần chạy (split, subset); tầng 2: bệnh nhân test_T phân tầng theo lớp, một lần mỗi split mỗi vòng (QĐ69a)",
           "seed_rule": "CRC32('rq2|cohort|n_t|comparison|k...')", "equality_rows_ok": n_gate, "results_opened_at_utc": opened,
           "r_input": str(args.r_input) if args.r_input else None,
           "r_input_sha256": sha256(args.r_input) if args.r_input and Path(args.r_input).suffix == ".zip" else None,
           "r_arms": list(R_ARMS) if r_data else [], "r_family": R_FAMILY if r_data else None,
           "elapsed_s": time.time() - t0, "versions": all_versions()}
    files = [p for p in (src, args.r_input) if p and Path(p).suffix == ".zip"]
    write_config(out, prov, cfg, data_files=files)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(tables["decisions"].to_string(index=False))
    print("ghi", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
