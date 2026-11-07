"""Độ nhạy của RQ1, phần (a) của QĐ53: sinh các dòng độ nhạy từ kết quả chính thức, có provenance. Không chạy lại mô hình, không đụng bảng survey.

Cấu hình: main (3 cohort GPL570), four_cohorts (thêm GSE42127), no_ihc (bỏ 4 gen IHC, E1_classification_noIHC). L1-LR so với B0, model = lr_l2, chuỗi k cố định
theo survey_v2.kstar. Kết luận H1 chính không đổi; độ nhạy chỉ báo cáo. Phần (b) của QĐ53 (BAS → LUSC, lọc F-test 2.000 gen) cần chạy E1 nên chưa nằm ở đây.

Đầu vào: <results-root>/E1_classification/*/classification_per_fold.csv (panels_results.zip), E1_classification_noIHC (ablation_phaseB_results.zip). Nếu có
tables/e1_kstar_pooled_by_k.csv thì đối chiếu cấu hình chính với bảng đó cho mọi phương pháp (92/92 dòng); lệch thì thoát mã 1 và không ghi đầu ra.

    python scripts/make_rq1_sensitivity.py

Đầu ra: results/survey_v2/RQ1_sensitivity/{rq1_sensitivity.csv, rq1_sensitivity_by_k.csv, provenance_manifest.json, config.json}.
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
from survey_v2.kstar import k_star, kstar_table  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402

METHOD = "l1_lr"
CLASSIFIER = "lr_l2"
GPL570 = tuple(C.POOLED_KSTAR_COHORTS)
CONFIGS = (("main", "E1_classification", GPL570),
           ("four_cohorts", "E1_classification", (*GPL570, "gse42127")),
           ("no_ihc", "E1_classification_noIHC", GPL570))
REFERENCE = ("tables", "e1_kstar_pooled_by_k.csv")
REFERENCE_TOLERANCE = 1e-9


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=None, help="mặc định $KLTN_DATA_ROOT/results/survey_v2")
    ap.add_argument("--out", default="RQ1_sensitivity")
    return ap.parse_args(argv)


def input_files(root: Path, folder: str) -> list[Path]:
    return sorted((root / folder).glob("*/classification_per_fold.csv"))


def load_per_fold(root: Path, folder: str) -> pd.DataFrame:
    files = input_files(root, folder)
    if not files:
        raise FileNotFoundError(f"không có classification_per_fold.csv dưới {root / folder}")
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return frame[frame.model == CLASSIFIER]


def _summary_row(name: str, cohorts, table: pd.DataFrame, k_ivw, k_both) -> dict:
    at_10 = table[table.k == 10].iloc[0]
    weights = {c: float(at_10[f"w_{c}"]) for c in cohorts}
    return {"config": name, "cohorts": ",".join(cohorts), "method": METHOD, "k_star_both": k_both, "k_star_ivw": k_ivw, "lb_ivw_k10": float(at_10.lb_ivw),
            "lb_eq_k10": float(at_10.lb_eq), "k_sequence": ">".join(str(int(k)) for k in table.k), "weights_k10": json.dumps(weights, sort_keys=True)}


def build(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(một dòng mỗi cấu hình, bảng theo k của L1-LR mỗi cấu hình, bảng theo k của mọi phương pháp ở cấu hình chính để đối chiếu)."""
    summary, by_k = [], []
    frames = {folder: load_per_fold(root, folder) for folder in {c[1] for c in CONFIGS}}
    for name, folder, cohorts in CONFIGS:
        sub = frames[folder]
        table, k_ivw, k_both = k_star(sub, METHOD, list(cohorts), require_all_cohorts=True)
        summary.append(_summary_row(name, cohorts, table, k_ivw, k_both))
        by_k.append(table.assign(config=name))
    main_frame = frames[CONFIGS[0][1]]
    methods = [m for m in main_frame.method.unique() if m not in ("b0", "random_k")]
    main_table, _ = kstar_table(main_frame, methods, cohorts=list(GPL570), require_all_cohorts=True)
    by_k = pd.concat(by_k, ignore_index=True)
    return pd.DataFrame(summary), by_k[["config", *[c for c in by_k.columns if c != "config"]]], main_table


def check_reference(table: pd.DataFrame, reference: pd.DataFrame) -> tuple[int, int, list]:
    """(số dòng khớp, tổng số dòng, các ô lệch (method, k, cột)). Khoá (method, k); dòng chỉ có ở một bên tính là không khớp, ghi cột "<row>"."""
    a, b = table.set_index(["method", "k"]), reference.assign(k=reference.k.astype(int)).set_index(["method", "k"])
    columns = [c for c in a.columns if c in b.columns and c != "missing_cohorts"]
    keys = a.index.union(b.index)
    n_ok, bad = 0, []
    for key in keys:
        if key not in a.index or key not in b.index:
            bad.append((*key, "<row>"))
            continue
        row_ok = True
        for col in columns:
            x, y = a.loc[key, col], b.loc[key, col]
            if pd.isna(x) and pd.isna(y):
                continue
            same = (not pd.isna(x) and not pd.isna(y)) and (bool(x) == bool(y) if isinstance(x, (bool, np.bool_)) else abs(float(x) - float(y)) <= REFERENCE_TOLERANCE)
            if not same:
                bad.append((*key, col))
                row_ok = False
        n_ok += row_ok
    return n_ok, len(keys), bad


def main(argv=None, script=__file__) -> int:
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t0 = time.time()
    root = args.results_root or DATA_ROOT.joinpath(*C.RESULT_DIR)
    summary, by_k, main_table = build(root)
    reference_path = root.joinpath(*REFERENCE)
    check = None
    if reference_path.exists():
        n_ok, n_all, bad = check_reference(main_table, pd.read_csv(reference_path, encoding="utf-8-sig"))
        check = {"file": str(reference_path), "sha256": sha256(reference_path), "rows_matched": n_ok, "rows_total": n_all}
        print(f"đối chiếu {reference_path.name}: {n_ok}/{n_all} dòng khớp", flush=True)
        if bad:
            print("LỆCH, không ghi đầu ra:", bad[:20], flush=True)
            return 1
    out = results_dir(args.out)
    summary.to_csv(out / "rq1_sensitivity.csv", index=False, encoding="utf-8-sig")
    by_k.to_csv(out / "rq1_sensitivity_by_k.csv", index=False, encoding="utf-8-sig")
    inputs = [f for folder in {c[1] for c in CONFIGS} for f in input_files(root, folder)]
    (out / "provenance_manifest.json").write_text(json.dumps(
        {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "script_commit": prov.get("commit_hash"), "inputs": {str(f): {"sha256": sha256(f)} for f in sorted(inputs)},
         "outputs": ["rq1_sensitivity.csv", "rq1_sensitivity_by_k.csv"], "reference_check": check}, ensure_ascii=False, indent=2), encoding="utf-8")
    cfg = {"experiment": "RQ1_sensitivity", "configs": [{"name": n, "input": f, "cohorts": list(c)} for n, f, c in CONFIGS], "method": METHOD, "classifier": CLASSIFIER,
           "reference_check": check, "note": "phần (a) của QĐ53; không chạy lại mô hình", "elapsed_s": time.time() - t0, "versions": all_versions()}
    write_config(out, prov, cfg)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.drop(columns="weights_k10").round(4).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
