"""Độ nhạy của RQ1, phần (a) của QĐ53: sinh các dòng độ nhạy từ kết quả chính thức, có provenance. Không chạy lại mô hình, không đụng bảng survey.

Cấu hình: main (3 cohort GPL570), four_cohorts (thêm GSE42127), no_ihc (bỏ 4 gen IHC, E1_classification_noIHC). L1-LR so với B0, model = lr_l2, chuỗi k cố định
theo survey_v2.kstar. Kết luận H1 chính không đổi; độ nhạy chỉ báo cáo.

Phần (b) của QĐ53 (Review 08/11 Mục 8.5), từ output của việc Kaggle rq1_sens_b; tự thêm khi các thư mục đã có (--require-part-b: bắt buộc):
  bas_lusc   gse30219 thay bằng gse30219_bas (E1_classification_bas: L1-LR và B0 trên bản chia 5×5 mới của biến thể); ghi tỉ lệ BAS trong
             LUSC (config.json của dữ liệu gse30219_bas, --bas-data-dir).
  ftest2000  L1-LR chọn trong 2.000 gen F-test của train (E1_classification_ftest2000) so với B0 của E1_classification chính (cùng bản chia,
             cùng seed, B0 không qua lọc nên không đổi).

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
# (tên, cohort, [(thư mục, phương pháp lấy từ thư mục đó hoặc None = mọi phương pháp)])
CONFIGS = (("main", GPL570, (("E1_classification", None),)),
           ("four_cohorts", (*GPL570, "gse42127"), (("E1_classification", None),)),
           ("no_ihc", GPL570, (("E1_classification_noIHC", None),)))
BAS = "gse30219_bas"
CONFIGS_B = (("bas_lusc", tuple(BAS if c == "gse30219" else c for c in GPL570), (("E1_classification", None), ("E1_classification_bas", None))),
             ("ftest2000", GPL570, (("E1_classification", ("b0",)), ("E1_classification_ftest2000", (METHOD,)))))
WHO_NOTE = "basaloid squamous cell carcinoma thuộc nhóm carcinoma tế bào vảy theo phân loại WHO 2015 (Travis2015)"
REFERENCE = ("tables", "e1_kstar_pooled_by_k.csv")
REFERENCE_TOLERANCE = 1e-9


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=None, help="mặc định $KLTN_DATA_ROOT/results/survey_v2")
    ap.add_argument("--out", default="RQ1_sensitivity")
    ap.add_argument("--require-part-b", action="store_true", help="dừng nếu chưa có output phần (b) (rq1_sens_b)")
    ap.add_argument("--bas-data-dir", type=Path, default=None, help="thư mục dữ liệu gse30219_bas (đọc config.json để lấy tỉ lệ BAS)")
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


def part_b_available(root: Path) -> bool:
    return all(input_files(root, folder) for _, _, parts in CONFIGS_B for folder, _ in parts)


def active_configs(root: Path, require_part_b: bool = False) -> tuple:
    if part_b_available(root):
        return CONFIGS + CONFIGS_B
    if require_part_b:
        missing = [f for _, _, parts in CONFIGS_B for f, _ in parts if not input_files(root, f)]
        raise FileNotFoundError(f"chưa có output phần (b): {sorted(set(missing))} dưới {root} (việc Kaggle rq1_sens_b)")
    return CONFIGS


def _frame(frames: dict, parts) -> pd.DataFrame:
    out = []
    for folder, methods in parts:
        f = frames[folder]
        out.append(f if methods is None else f[f.method.isin(methods)])
    frame = pd.concat(out, ignore_index=True)
    if frame.duplicated(["cohort", "repeat", "fold", "method", "k", "model"]).any():
        raise ValueError(f"trùng dòng khi ghép {[p[0] for p in parts]}")
    return frame


def build(root: Path, require_part_b: bool = False, bas_note: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(một dòng mỗi cấu hình, bảng theo k của L1-LR mỗi cấu hình, bảng theo k của mọi phương pháp ở cấu hình chính để đối chiếu)."""
    summary, by_k = [], []
    configs = active_configs(root, require_part_b)
    frames = {folder: load_per_fold(root, folder) for _, _, parts in configs for folder, _ in parts}
    for name, cohorts, parts in configs:
        sub = _frame(frames, parts)
        table, k_ivw, k_both = k_star(sub, METHOD, list(cohorts), require_all_cohorts=True)
        row = _summary_row(name, cohorts, table, k_ivw, k_both)
        row["note"] = bas_note if name == "bas_lusc" else None
        summary.append(row)
        by_k.append(table.assign(config=name))
    main_frame = frames[CONFIGS[0][2][0][0]]
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
    bas_note = None
    bas_cfg_path = (args.bas_data_dir or DATA_ROOT / C.GEO_TARGET_DIR / BAS) / "config.json"
    if bas_cfg_path.exists():
        bc = json.loads(bas_cfg_path.read_text(encoding="utf-8"))
        bas_note = (f"BAS {bc['n_bas_kept']}/{bc['n_per_class'][1]} mẫu LUSC ({bc['bas_fraction_of_lusc']:.1%}); n = {bc['n_per_class'][0]}/{bc['n_per_class'][1]}; "
                    + WHO_NOTE)
    summary, by_k, main_table = build(root, args.require_part_b, bas_note)
    configs = active_configs(root, args.require_part_b)
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
    inputs = [f for folder in {p[0] for _, _, parts in configs for p in parts} for f in input_files(root, folder)]
    if bas_cfg_path.exists():
        inputs.append(bas_cfg_path)
    (out / "provenance_manifest.json").write_text(json.dumps(
        {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "script_commit": prov.get("commit_hash"), "inputs": {str(f): {"sha256": sha256(f)} for f in sorted(inputs)},
         "outputs": ["rq1_sensitivity.csv", "rq1_sensitivity_by_k.csv"], "reference_check": check}, ensure_ascii=False, indent=2), encoding="utf-8")
    cfg = {"experiment": "RQ1_sensitivity", "configs": [{"name": n, "cohorts": list(c), "inputs": [{"folder": f, "methods": list(m) if m else None} for f, m in parts]}
                                                       for n, c, parts in configs],
           "part_b_included": len(configs) > len(CONFIGS), "bas_note": bas_note, "bas_data_config": str(bas_cfg_path) if bas_cfg_path.exists() else None,
           "method": METHOD, "classifier": CLASSIFIER, "reference_check": check,
           "note": "QĐ53 phần (a) và, khi có output rq1_sens_b, phần (b); không chạy lại mô hình", "elapsed_s": time.time() - t0, "versions": all_versions()}
    write_config(out, prov, cfg)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.drop(columns="weights_k10").round(4).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
