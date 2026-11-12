"""W1 (KBTN v2.6, QĐ125 Review 11/11): gắn nhãn đọc QĐ97e lên bảng so sánh mô tả MVP với ptLasso, Trans-GLM đã sinh 09/11
(scripts/make_rq2_tables.py tại 2b971e8, rq2_tables_2b971e8_results.zip). KHÔNG đổi số, không chạy lại bootstrap: mọi cột của
rq2_baselines.csv được chép nguyên, chỉ thêm cột nhãn và cột tỉ lệ same_as_stage1_topk.

Quy tắc QĐ97e (Review 09/11 Mục 6.4e, KBTN v2.6 Mục 9), nguyên văn: Δ = MVP − baseline, trung bình k ∈ {2, 3, 5}, CI 95 % hai phía theo
IVW và trọng số bằng nhau, từng n_T. "MVP higher" chỉ khi cận dưới > 0 ở cả hai cách gộp; "equivalent within ±0.01" khi cả hai CI nằm
trong (−0,01; 0,01); "baseline higher" khi cận trên < 0 ở cả hai; còn lại "inconclusive". Không Holm; chú thích ghi rõ là so sánh mô tả;
báo same_as_stage1_topk cạnh mọi dòng ptLasso. Chỉ bốn nhãn đó (QĐ125).

LỰA CHỌN CÀI ĐẶT (quy tắc chưa nêu, ghi vào config.json để GVHD xác nhận):
  - Một dòng có thể thoả cả "MVP higher" (hoặc "baseline higher") và "equivalent within ±0.01" khi CI nằm trọn trong (0; 0,01). Khi đó
    nhãn theo thứ tự quy tắc viết ("MVP higher", rồi "equivalent", rồi "baseline higher") và cột also_equivalent = True.
  - CI không xác định (IVW NaN khi một cohort có SE = 0) thì điều kiện không thoả, tức "inconclusive"; cột ci_undefined đánh dấu.
  - same_as_stage1_topk_rate cạnh dòng ptLasso: trung bình các tỉ lệ của rq2_ptlasso_degeneracy.csv trên các cohort của bộ gộp và
    k ∈ {2, 3, 5}, trọng số n_defined; kèm khoảng nhỏ nhất – lớn nhất.

    python scripts/make_w1_r_labels.py --rq2-tables rq2_tables_2b971e8_results.zip

Đầu ra: results/survey_v2/RQ2_r_labels/{rq2_baselines_labeled.csv, rq2_baselines_caption.txt, config.json}.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2.data import sha256  # noqa: E402
from survey_v2.runlog import all_versions, results_dir, start_run, write_config  # noqa: E402

K_AVG = (2, 3, 5)
EQ_MARGIN = 0.01
LABELS = ("MVP higher", "equivalent within ±0.01", "baseline higher", "inconclusive")
FAMILY = "so sánh mô tả (QĐ97e; không đăng ký giả thuyết; không Holm)"
FILES = ("rq2_baselines.csv", "rq2_ptlasso_degeneracy.csv", "config.json")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq2-tables", type=Path, required=True, help="zip bảng RQ2 có rq2_baselines.csv (2b971e8) hoặc thư mục RQ2_tables")
    ap.add_argument("--out", default="RQ2_r_labels")
    return ap.parse_args(argv)


def load(src: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, bytes]:
    """(rq2_baselines, rq2_ptlasso_degeneracy, config của lần sinh bảng, byte gốc của rq2_baselines.csv)."""
    src = Path(src)
    raw = {}
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as zf:
            for name in FILES:
                hit = [n for n in zf.namelist() if n.endswith("RQ2_tables/" + name)]
                if len(hit) != 1:
                    raise SystemExit(f"{src}: cần đúng một RQ2_tables/{name}, thấy {len(hit)}")
                raw[name] = zf.read(hit[0])
    else:
        for name in FILES:
            raw[name] = (src / name).read_bytes()
    read = lambda b: pd.read_csv(io.BytesIO(b), dtype={"n_t": str}, encoding="utf-8-sig", float_precision="round_trip")  # noqa: E731
    return read(raw[FILES[0]]), read(raw[FILES[1]]), json.loads(raw[FILES[2]]), raw[FILES[0]]


def label_row(r) -> dict:
    lo = np.array([r.ci95_ivw_low, r.ci95_eq_low], float)
    hi = np.array([r.ci95_ivw_high, r.ci95_eq_high], float)
    undefined = not np.isfinite(np.r_[lo, hi]).all()
    higher = (not undefined) and bool((lo > 0).all())
    lower = (not undefined) and bool((hi < 0).all())
    equiv = (not undefined) and bool((lo > -EQ_MARGIN).all() and (hi < EQ_MARGIN).all())
    label = LABELS[0] if higher else LABELS[1] if equiv else LABELS[2] if lower else LABELS[3]
    return {"label": label, "rule_mvp_higher": higher, "rule_equivalent": equiv, "rule_baseline_higher": lower,
            "also_equivalent": equiv and label != LABELS[1], "ci_undefined": undefined}


def stage1_rate(deg: pd.DataFrame, cohorts: list, n_t: str) -> dict:
    d = deg[deg.cohort.isin(cohorts) & (deg.n_t.astype(str) == n_t) & deg.k.isin(K_AVG)].dropna(subset=["same_as_stage1_topk_rate"])
    if d.empty or d.n_defined.sum() == 0:
        return {"same_as_stage1_topk_rate": np.nan, "same_as_stage1_topk_min": np.nan, "same_as_stage1_topk_max": np.nan, "n_defined": 0}
    w = d.n_defined.to_numpy(float)
    v = d.same_as_stage1_topk_rate.to_numpy(float)
    return {"same_as_stage1_topk_rate": float((w * v).sum() / w.sum()), "same_as_stage1_topk_min": float(v.min()),
            "same_as_stage1_topk_max": float(v.max()), "n_defined": int(w.sum())}


def label_table(base: pd.DataFrame, deg: pd.DataFrame) -> pd.DataFrame:
    extra = []
    for r in base.itertuples(index=False):
        row = label_row(r)
        if r.comparison == "ptlasso":
            row.update(stage1_rate(deg, str(r.cohorts).split(), str(r.n_t)))
        else:
            row.update({"same_as_stage1_topk_rate": np.nan, "same_as_stage1_topk_min": np.nan, "same_as_stage1_topk_max": np.nan, "n_defined": np.nan})
        extra.append(row)
    out = pd.concat([base.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    out["reading_family"] = FAMILY
    return out


def caption(cfg: dict, commit: str | None) -> str:
    return (f"So sánh mô tả MVP (m5_pos) với ptLasso và Trans-GLM: Δ = MVP − baseline, AUROC trung bình k ∈ {{2, 3, 5}}, CI 95 % hai phía "
            f"theo IVW và trọng số bằng nhau; không Holm, không đăng ký giả thuyết. Nhãn đọc theo QĐ97e. Thứ tự thời gian: bảng sinh "
            f"{cfg.get('results_opened_at_utc')} (make_rq2_tables.py tại {str(cfg.get('commit_hash'))[:7]}); quy tắc đọc chốt cùng ngày ở "
            f"Review 09/11, không dựa vào số của bảng; số không đổi. Nhãn gán bằng make_w1_r_labels.py tại {str(commit)[:7]}. "
            f"same_as_stage1_topk: tỉ lệ lần chạy ptLasso có panel giai đoạn 2 trùng top-k của giai đoạn 1, báo cạnh mọi dòng ptLasso.")


def main(argv=None, script=__file__) -> int:
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t0 = time.time()
    base, deg, src_cfg, raw = load(args.rq2_tables)
    table = label_table(base, deg)
    if not table[list(base.columns)].equals(base):
        raise SystemExit("cột gốc của rq2_baselines.csv bị đổi")
    out = results_dir(args.out)
    table.to_csv(out / "rq2_baselines_labeled.csv", index=False, encoding="utf-8-sig")
    cap = caption(src_cfg, prov.get("commit_hash"))
    (out / "rq2_baselines_caption.txt").write_text(cap + "\n", encoding="utf-8")
    files = [args.rq2_tables] if args.rq2_tables.suffix == ".zip" else []
    cfg = {"experiment": "RQ2_r_labels", "rq2_tables_input": str(args.rq2_tables),
           "input_sha256": {p.name: sha256(p) for p in files}, "rq2_baselines_sha256": hashlib.sha256(raw).hexdigest(),
           "source_table_commit": src_cfg.get("commit_hash"), "source_table_opened_at_utc": src_cfg.get("results_opened_at_utc"),
           "source_traceable": src_cfg.get("traceable"), "labels": list(LABELS), "eq_margin": EQ_MARGIN, "k_avg": list(K_AVG), "family": FAMILY,
           "choices_to_confirm": [
               "dòng thoả cả 'MVP higher'/'baseline higher' và 'equivalent within ±0.01': nhãn theo thứ tự quy tắc viết, cột also_equivalent",
               "CI không xác định: 'inconclusive', cột ci_undefined",
               "same_as_stage1_topk_rate: trung bình trọng số n_defined trên cohort của bộ gộp và k ∈ {2, 3, 5}, kèm min – max"],
           "label_counts": table.groupby(["pool_set", "label"]).size().rename("n").reset_index().to_dict("records"),
           "elapsed_s": time.time() - t0, "versions": all_versions()}
    write_config(out, prov, cfg, data_files=files)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table[["comparison", "n_t", "pool_set", "ci95_ivw_low", "ci95_ivw_high", "ci95_eq_low", "ci95_eq_high", "label",
                     "also_equivalent", "same_as_stage1_topk_rate"]].round(4).to_string(index=False))
    print("ghi", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
