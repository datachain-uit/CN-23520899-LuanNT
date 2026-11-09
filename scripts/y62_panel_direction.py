"""Y62 phần 1 (Review 07/11 Y62; Review 08/11 Mục 8.7, QĐ71a): chiều LUSC − LUAD của các gen panel nguồn E7 trên từng nền tảng.

Panel nguồn đọc từ config.json chính thức của E7 (trường source_panels, mọi cohort phải giống nhau và lồng nhau theo k):
k = 2 {DSG3, KRT5}, k = 3 + DSC3, k = 4 + HNF1B, k = 5 + ARHGEF38. Với từng bộ (TCGA và 4 cohort GEO, dữ liệu sau QC, không gian gen
như load_dataset) và từng gen: trung bình hai lớp trên log2 gốc, hiệu LUSC − LUAD, t Welch, p hai phía, lớp có trung bình cao hơn.
Với cặp DSG3/KRT5: số và tỉ lệ mẫu có DSG3 > KRT5 theo lớp; AUROC của hiệu DSG3 − KRT5 khi dự đoán LUSC. Ở k = 2, chuẩn hoá trong mẫu
φ của E7 chỉ giữ dấu của hiệu này, nên AUROC < 0,5 nghĩa là quy tắc "DSG3 > KRT5 → LUSC" đảo chiều trên bộ đó. Đây là thống kê mô tả,
không phải tái lập AUROC zero-shot của E7. Không huấn luyện mô hình.

    python scripts/y62_panel_direction.py --e7 e7_results.zip

Đầu ra: results/survey_v2/Y62_panel_direction/{y62_gene_direction.csv, y62_dsg3_vs_krt5.csv, config.json}.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import load_dataset, sha256  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402

K_MAX = 5
PAIR = ("DSG3", "KRT5")
CLASS_NAME = {0: "LUAD", 1: "LUSC"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e7", type=Path, default=None, help="zip của việc e7 hoặc thư mục E7 (mặc định $KLTN_DATA_ROOT/results/survey_v2/E7)")
    ap.add_argument("--datasets", nargs="+", default=[C.SOURCE_DATASET, *C.FEWSHOT_COHORTS])
    ap.add_argument("--k-max", type=int, default=K_MAX)
    ap.add_argument("--out", default="Y62_panel_direction")
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------------------------- panel nguồn E7
def _read_e7_configs(src: Path, cohorts) -> dict:
    src = Path(src)
    out = {}
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as zf:
            names = zf.namelist()
            for c in cohorts:
                hit = [n for n in names if n.endswith(f"E7/{c}/config.json")]
                if len(hit) != 1:
                    raise ValueError(f"{src}: cần đúng một E7/{c}/config.json, thấy {len(hit)}")
                out[c] = json.loads(zf.read(hit[0]))
    else:
        for c in cohorts:
            out[c] = json.loads((src / c / "config.json").read_text(encoding="utf-8"))
    return out


def panel_entry_k(src: Path, cohorts, k_max: int = K_MAX) -> tuple[dict, dict]:
    """{gen: k nhỏ nhất có gen trong panel nguồn} cho k <= k_max; panel phải giống nhau ở mọi cohort và lồng nhau theo k."""
    cfgs = _read_e7_configs(src, cohorts)
    panels = {c: {int(k): sorted(v) for k, v in cfg["source_panels"].items() if int(k) <= k_max} for c, cfg in cfgs.items()}
    first = next(iter(panels.values()))
    if any(p != first for p in panels.values()):
        raise ValueError(f"panel nguồn của E7 khác nhau giữa các cohort: {panels}")
    entry, prev = {}, set()
    for k in sorted(first):
        cur = set(first[k])
        if not prev <= cur:
            raise ValueError(f"panel nguồn không lồng nhau ở k = {k}: {sorted(prev - cur)} biến mất")
        for g in sorted(cur - prev):
            entry[g] = k
        prev = cur
    meta = {"source_panels": {str(k): v for k, v in first.items()}, "commit_hash": {c: cfg.get("commit_hash") for c, cfg in cfgs.items()}}
    return entry, meta


# ------------------------------------------------------------------------------------------------- thống kê
def gene_direction(frames: dict, entry: dict) -> pd.DataFrame:
    """frames = {bộ: (X DataFrame mẫu × gen, y 0/1, nền tảng)}. Một dòng cho mỗi (bộ, gen); gen thiếu ghi present = False."""
    rows = []
    for name, (X, y, platform) in frames.items():
        y = np.asarray(y, int)
        for gene, k in sorted(entry.items(), key=lambda t: (t[1], t[0])):
            row = {"dataset": name, "platform": platform, "gene": gene, "k_enter": k, "n_luad": int((y == 0).sum()), "n_lusc": int((y == 1).sum()),
                   "present": gene in X.columns}
            if row["present"]:
                v = X[gene].to_numpy(float)
                a, b = v[y == 1], v[y == 0]
                t = stats.ttest_ind(a, b, equal_var=False)
                row.update({"mean_luad": float(b.mean()), "mean_lusc": float(a.mean()), "diff_lusc_minus_luad": float(a.mean() - b.mean()),
                            "welch_t": float(t.statistic), "p_value": float(t.pvalue), "higher_in": "LUSC" if t.statistic > 0 else "LUAD"})
            else:
                row.update({"mean_luad": np.nan, "mean_lusc": np.nan, "diff_lusc_minus_luad": np.nan, "welch_t": np.nan, "p_value": np.nan, "higher_in": None})
            rows.append(row)
    return pd.DataFrame(rows)


def pair_table(frames: dict, a: str = PAIR[0], b: str = PAIR[1]) -> pd.DataFrame:
    """Theo bộ và lớp: số / tỉ lệ mẫu có a > b; dòng class = all có AUROC của (a − b) khi dự đoán LUSC."""
    rows = []
    for name, (X, y, platform) in frames.items():
        y = np.asarray(y, int)
        if a not in X.columns or b not in X.columns:
            rows.append({"dataset": name, "platform": platform, "class": "all", "n": len(y), "n_a_gt_b": None, "frac_a_gt_b": None,
                         "mean_a_minus_b": None, "auroc_a_minus_b": None, "note": f"thiếu {a} hoặc {b}"})
            continue
        d = X[a].to_numpy(float) - X[b].to_numpy(float)
        for lab in (0, 1):
            m = y == lab
            rows.append({"dataset": name, "platform": platform, "class": CLASS_NAME[lab], "n": int(m.sum()), "n_a_gt_b": int((d[m] > 0).sum()),
                         "frac_a_gt_b": float((d[m] > 0).mean()), "mean_a_minus_b": float(d[m].mean()), "auroc_a_minus_b": None, "note": None})
        rows.append({"dataset": name, "platform": platform, "class": "all", "n": len(y), "n_a_gt_b": int((d > 0).sum()), "frac_a_gt_b": float((d > 0).mean()),
                     "mean_a_minus_b": float(d.mean()), "auroc_a_minus_b": float(roc_auc_score(y, d)), "note": None})
    return pd.DataFrame(rows)


def build(frames: dict, entry: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    return gene_direction(frames, entry), pair_table(frames)


def load_frames(names) -> tuple[dict, dict]:
    frames, files = {}, {}
    for name in names:
        ds = load_dataset(DATA_ROOT, name)
        frames[name] = (pd.DataFrame(ds.X, index=ds.sample_ids, columns=ds.genes), ds.y, C.DATASET_PLATFORM.get(name, ""))
        files[name] = {"expression.h5": sha256(ds.data_dir / "expression.h5"), "metadata.csv": sha256(ds.data_dir / "metadata.csv"),
                       "n_per_class": [int((ds.y == 0).sum()), int((ds.y == 1).sum())], "n_genes": int(ds.p)}
    return frames, files


def main(argv=None, script=__file__) -> int:
    args = parse_args(argv)
    e7 = args.e7 or DATA_ROOT.joinpath(*C.RESULT_DIR, "E7")
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t0 = time.time()
    entry, meta = panel_entry_k(e7, [d for d in args.datasets if d != C.SOURCE_DATASET], args.k_max)
    frames, files = load_frames(args.datasets)
    direction, pairs = build(frames, entry)
    out = results_dir(args.out)
    direction.to_csv(out / "y62_gene_direction.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(out / "y62_dsg3_vs_krt5.csv", index=False, encoding="utf-8-sig")
    cfg = {"experiment": "Y62_panel_direction", "e7_input": str(e7), "e7_input_sha256": sha256(e7) if Path(e7).suffix == ".zip" else None,
           "e7_source_panels": meta["source_panels"], "e7_commit_hash": meta["commit_hash"], "gene_entry_k": entry, "pair": list(PAIR),
           "datasets": files, "statistic": "Welch t (scipy.stats.ttest_ind, equal_var=False), LUSC − LUAD trên log2 gốc của bộ (không z-score)",
           "pair_note": "AUROC của DSG3 − KRT5 dự đoán LUSC; φ của E7 ở k = 2 chỉ giữ dấu của hiệu này; mô tả, không tái lập E7",
           "elapsed_s": time.time() - t0, "versions": all_versions()}
    write_config(out, prov, cfg, data_files=[e7] if Path(e7).suffix == ".zip" else ())
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(direction.pivot(index=["k_enter", "gene"], columns="dataset", values="welch_t").round(2).to_string())
        print(pairs.round(3).to_string(index=False))
    print("ghi", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
