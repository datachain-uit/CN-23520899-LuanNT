"""Cổng bằng nhau của RQ2 (QĐ51; Review 07/11 Bảng 7 bước 5): chạy trước khi tính thống kê.

So theo khoá (split, subset, n_t, k), năm chỉ số AUROC, BAcc, MCC, log-loss, Brier, sai khác tối đa 10⁻¹²:
* target_only: dòng MVP γ = 0, gamma_rule = fixed, prefilter_space = target của RQ2 với l1_lr của e1b_per_run.csv;
* ihc_2, ihc_4: với cùng method của e1b_per_run.csv;
* select_on_source, pooled: với cùng strategy của e7_per_run.csv.

Mọi dòng đối chứng có k trong lưới của RQ2 phải có mặt ở RQ2, nên phiên chạy dở không qua được cổng; --n-t chỉ dùng khi cố ý giới hạn mức n_T.
Không đạt: thoát mã 1, ghi rq2_equality_report.csv để xem dòng nào lệch; dừng và báo cáo, không sửa tay (QĐ51).

    python scripts/check_rq2_equality.py

Đầu ra: <results-root>/<rq2-dir>/rq2_equality_report.csv, mỗi dòng một khoá với status ∈ {ok, diff, missing_in_rq2, missing_in_reference, duplicate, missing_file}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402

METRICS = ("auroc", "bacc", "mcc", "logloss", "brier")
KEY = ["split", "subset", "n_t", "k"]
TOLERANCE = 1e-12
REPORT = "rq2_equality_report.csv"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=None, help="mặc định $KLTN_DATA_ROOT/results/survey_v2")
    ap.add_argument("--datasets", nargs="+", default=list(C.FEWSHOT_COHORTS))
    ap.add_argument("--rq2-dir", default="RQ2_mvp")
    ap.add_argument("--e1b-dir", default="E1b")
    ap.add_argument("--e7-dir", default="E7")
    ap.add_argument("--n-t", nargs="+", default=None, help="chỉ so các mức n_T này; mặc định mọi mức có trong đối chứng")
    return ap.parse_args(argv)


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, dtype={"n_t": str}, float_precision="round_trip") if path.exists() else None


def _worst_difference(row: pd.Series) -> tuple[float, str]:
    """Sai khác lớn nhất của một khoá. Trống ở cả hai phía là bằng nhau, trống ở một phía là khác."""
    worst, name = 0.0, ""
    for metric in METRICS:
        a, b = row[f"{metric}_rq2"], row[f"{metric}_ref"]
        if np.isnan(a) and np.isnan(b):
            continue
        d = float("inf") if np.isnan(a) or np.isnan(b) else abs(float(a) - float(b))
        if d > worst:
            worst, name = d, metric
    return worst, name


def compare_arm(cohort: str, arm: str, reference: str, own: pd.DataFrame, ref: pd.DataFrame) -> list[dict]:
    base = {"cohort": cohort, "arm": arm, "reference": reference}
    rows = []
    for side, frame in (("rq2", own), ("reference", ref)):
        for key in frame.loc[frame.duplicated(KEY), KEY].drop_duplicates().itertuples(index=False, name=None):
            rows.append({**base, **dict(zip(KEY, key)), "max_abs_diff": np.nan, "worst_metric": "", "status": "duplicate", "note": side})
    own, ref = own.drop_duplicates(KEY), ref.drop_duplicates(KEY)
    merged = own[KEY + list(METRICS)].merge(ref[KEY + list(METRICS)], on=KEY, how="outer", suffixes=("_rq2", "_ref"), indicator=True)
    for _, row in merged.sort_values(KEY).iterrows():
        key = {c: row[c] for c in KEY}
        if row["_merge"] != "both":
            status = "missing_in_rq2" if row["_merge"] == "right_only" else "missing_in_reference"
            rows.append({**base, **key, "max_abs_diff": np.nan, "worst_metric": "", "status": status, "note": ""})
            continue
        worst, name = _worst_difference(row)
        rows.append({**base, **key, "max_abs_diff": worst, "worst_metric": name, "status": "diff" if worst > TOLERANCE else "ok", "note": ""})
    return rows


def compare_cohort(cohort: str, rq2: pd.DataFrame | None, e1b: pd.DataFrame | None, e7: pd.DataFrame | None, n_t) -> list[dict]:
    missing = [name for name, frame in (("rq2", rq2), ("e1b", e1b), ("e7", e7)) if frame is None]
    if missing:
        return [{"cohort": cohort, "arm": "", "reference": "", "status": "missing_file", "note": name} for name in missing]
    if n_t:
        rq2, e1b, e7 = rq2[rq2.n_t.isin(n_t)], e1b[e1b.n_t.isin(n_t)], e7[e7.n_t.isin(n_t)]
    mvp = rq2[rq2.method == "mvp"]
    k_grid = set(mvp.k)
    target = mvp[(mvp.gamma_rule == "fixed") & (mvp.gamma == 0) & (mvp.prefilter_space == "target")]
    rows = compare_arm(cohort, "target_only", "e1b", target, e1b[(e1b.method == "l1_lr") & e1b.k.isin(k_grid)])
    for method in ("ihc_2", "ihc_4"):
        rows += compare_arm(cohort, method, "e1b", rq2[rq2.method == method], e1b[e1b.method == method])
    for method in ("select_on_source", "pooled"):
        rows += compare_arm(cohort, method, "e7", rq2[rq2.method == method], e7[(e7.strategy == method) & e7.k.isin(k_grid)])
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.results_root is None:
        from survey_v2.runlog import DATA_ROOT
        args.results_root = DATA_ROOT.joinpath(*C.RESULT_DIR)
    root = args.results_root
    rows = []
    for name in args.datasets:
        rows += compare_cohort(name, _read(root / args.rq2_dir / name / "rq2_mvp_per_run.csv"), _read(root / args.e1b_dir / name / "e1b_per_run.csv"),
                               _read(root / args.e7_dir / name / "e7_per_run.csv"), args.n_t)
    columns = ["cohort", "arm", "reference", *KEY, "max_abs_diff", "worst_metric", "status", "note"]
    report = pd.DataFrame(rows).reindex(columns=columns)
    (root / args.rq2_dir).mkdir(parents=True, exist_ok=True)
    report.to_csv(root / args.rq2_dir / REPORT, index=False, encoding="utf-8-sig")
    counts = report.status.value_counts().to_dict()
    print(f"cổng bằng nhau: {counts}; báo cáo: {root / args.rq2_dir / REPORT}", flush=True)
    passed = len(report) > 0 and set(counts) == {"ok"}
    print("EQUALITY_OK" if passed else "EQUALITY_FAILED: dừng và báo cáo, không sửa tay (QĐ51)", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
