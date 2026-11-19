"""W5 (KBTN v2.6; Review 09/11 V4, Y81, QĐ96): mẫu BAS của GSE30219 gần tâm SQC (LUSC) hay tâm ADC (LUAD).

Mô tả hỗ trợ cho ánh xạ BAS → LUSC của biến thể gse30219_bas (cạnh căn cứ WHO 2015, Travis2015). Chỉ dùng ở chương dữ liệu; không
dùng làm bằng chứng cho H1 hay để chọn cấu hình (QĐ96, V4). Không huấn luyện mô hình, không dùng nhãn khi chọn gen.

Cách tính, trên dữ liệu gse30219_bas sau QC (180 mẫu: 81 ADC, 60 SQC, 39 BAS; log2 gốc, không gian gen như load_dataset):
  1. Gen: n gen có phương sai lớn nhất trên toàn bộ 180 mẫu (không dùng nhãn hay histology); chính n = 2.000, phụ 1.000 và 5.000.
  2. Mỗi gen trừ trung bình của nó trên 180 mẫu (không dùng nhãn), để tương quan với tâm không bị mức biểu hiện chung của gen chi phối.
  3. Tâm ADC = trung bình các mẫu ADC; tâm SQC = trung bình các mẫu SQC. Mẫu BAS KHÔNG vào tâm nào. Với chính một mẫu ADC hoặc SQC,
     tâm của nhóm nó được tính lại không có mẫu đó (leave-one-out), để ba nhóm so được với nhau.
  4. Mỗi mẫu: d_r = r(x, tâm SQC) − r(x, tâm ADC) (Pearson); d_e = ‖x − tâm ADC‖ − ‖x − tâm SQC‖ (Euclid). Hai hiệu > 0 nghĩa là
     mẫu gần SQC hơn.
Báo phân bố của d_r, d_e theo nhóm (n, số > 0, nhỏ nhất, tứ phân vị, trung vị, trung bình, lớn nhất), không chỉ số đếm (Y81), và trung
bình log2 của bốn gen IHC_4 (NKX2-1, NAPSA: ADC; TP63, KRT5: SQC) theo nhóm.

    python scripts/w5_bas_vs_sqc.py

Đầu ra: results/survey_v2/W5_bas_vs_sqc/{w5_per_sample.csv, w5_summary.csv, w5_ihc_markers.csv, config.json}.
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
from survey_v2.data import load_dataset, sha256  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402

DATASET = "gse30219_bas"
GROUPS = ("ADC", "SQC", "BAS")
N_GENES_MAIN = 2000
N_GENES_ALL = (1000, 2000, 5000)
QUANTILES = {"min": 0.0, "q25": 0.25, "median": 0.5, "q75": 0.75, "max": 1.0}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--n-genes", type=int, nargs="+", default=list(N_GENES_ALL))
    ap.add_argument("--out", default="W5_bas_vs_sqc")
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------------------------- tính
def top_variance_genes(X: np.ndarray, n: int) -> np.ndarray:
    """Chỉ số của n gen phương sai lớn nhất (ddof = 1) trên mọi mẫu; hoà thì giữ gen đứng trước. Không dùng nhãn."""
    if n > X.shape[1]:
        raise ValueError(f"n_genes = {n} lớn hơn số gen {X.shape[1]}")
    v = X.var(axis=0, ddof=1)
    return np.sort(np.argsort(-v, kind="stable")[:n])


def _corr(x: np.ndarray, c: np.ndarray) -> float:
    return float(np.corrcoef(x, c)[0, 1])


def per_sample(X: np.ndarray, histology: np.ndarray, sample_ids, n_genes: int) -> pd.DataFrame:
    """Một dòng mỗi mẫu: d_r, d_e trên n_genes gen biến thiên nhất, gen đã trừ trung bình; tâm leave-one-out cho ADC, SQC."""
    histology = np.asarray(histology)
    unknown = sorted(set(histology) - set(GROUPS))
    if unknown:
        raise ValueError(f"histology ngoài {GROUPS}: {unknown}")
    Z = X[:, top_variance_genes(X, n_genes)]
    Z = Z - Z.mean(axis=0)
    idx = {g: np.flatnonzero(histology == g) for g in ("ADC", "SQC")}
    sums = {g: Z[i].sum(axis=0) for g, i in idx.items()}
    rows = []
    for j, (sid, h) in enumerate(zip(sample_ids, histology)):
        cen = {}
        for g in ("ADC", "SQC"):
            n = len(idx[g])
            cen[g] = (sums[g] - Z[j]) / (n - 1) if h == g else sums[g] / n
        x = Z[j]
        r_sqc, r_adc = _corr(x, cen["SQC"]), _corr(x, cen["ADC"])
        e_sqc, e_adc = float(np.linalg.norm(x - cen["SQC"])), float(np.linalg.norm(x - cen["ADC"]))
        rows.append({"n_genes": n_genes, "sample_id": str(sid), "histology": h, "r_sqc": r_sqc, "r_adc": r_adc, "d_r": r_sqc - r_adc,
                     "dist_sqc": e_sqc, "dist_adc": e_adc, "d_e": e_adc - e_sqc, "closer_to": "SQC" if r_sqc > r_adc else "ADC"})
    return pd.DataFrame(rows)


def summarize(per: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n_genes, h), g in per.groupby(["n_genes", "histology"], sort=False):
        for metric in ("d_r", "d_e"):
            v = g[metric].to_numpy(float)
            row = {"n_genes": int(n_genes), "histology": h, "metric": metric, "n": len(v), "n_positive": int((v > 0).sum()),
                   "frac_positive": float((v > 0).mean()), "mean": float(v.mean())}
            row.update({k: float(np.quantile(v, q)) for k, q in QUANTILES.items()})
            rows.append(row)
    out = pd.DataFrame(rows)
    order = {h: i for i, h in enumerate(GROUPS)}
    return out.sort_values(["n_genes", "metric", "histology"], key=lambda s: s.map(order) if s.name == "histology" else s,
                           kind="stable").reset_index(drop=True)


def ihc_markers(X: pd.DataFrame, histology: np.ndarray) -> pd.DataFrame:
    rows = []
    for gene in C.IHC_4:
        row = {"gene": gene, "present": gene in X.columns}
        if row["present"]:
            for h in GROUPS:
                v = X.loc[np.asarray(histology) == h, gene].to_numpy(float)
                row[f"mean_{h}"] = float(v.mean())
                row[f"sd_{h}"] = float(v.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def read_histology(meta_path: Path, sample_ids) -> np.ndarray:
    meta = pd.read_csv(meta_path, usecols=["sample_id", "histology_raw"]).set_index("sample_id")
    h = meta["histology_raw"].astype(str).str.replace("histology:", "", regex=False).str.strip()
    return h.loc[[str(s) for s in sample_ids]].to_numpy()


# ------------------------------------------------------------------------------------------------- chạy
def main(argv=None, script=__file__) -> int:
    args = parse_args(argv)
    prov = start_run(script, params=vars(args))
    t0 = time.time()
    ds = load_dataset(DATA_ROOT, args.dataset)
    hist = read_histology(ds.data_dir / "metadata.csv", ds.sample_ids)
    if not np.array_equal(np.isin(hist, ("SQC", "BAS")).astype(int), np.asarray(ds.y, int)):
        raise ValueError("nhãn của load_dataset không khớp histology (SQC, BAS → LUSC = 1; ADC → LUAD = 0)")
    per = pd.concat([per_sample(ds.X, hist, ds.sample_ids, n) for n in args.n_genes], ignore_index=True)
    summary = summarize(per)
    ihc = ihc_markers(pd.DataFrame(ds.X, columns=ds.genes), hist)
    out = results_dir(args.out)
    per.to_csv(out / "w5_per_sample.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "w5_summary.csv", index=False, encoding="utf-8-sig")
    ihc.to_csv(out / "w5_ihc_markers.csv", index=False, encoding="utf-8-sig")
    counts = {h: int((hist == h).sum()) for h in GROUPS}
    cfg = {"experiment": "W5_bas_vs_sqc", "dataset": args.dataset, "n_genes_main": N_GENES_MAIN, "n_genes": list(args.n_genes),
           "histology_counts": counts, "n_genes_dataset": int(ds.p), "gene_space": ds.notes.get("gene_space"),
           "files_sha256": {"expression.h5": sha256(ds.data_dir / "expression.h5"), "metadata.csv": sha256(ds.data_dir / "metadata.csv")},
           "gene_selection": "n gen phương sai lớn nhất (ddof = 1) trên mọi mẫu, không dùng nhãn hay histology",
           "centering": "mỗi gen trừ trung bình trên mọi mẫu (không dùng nhãn)",
           "centroids": "ADC và SQC; BAS không vào tâm nào; mẫu ADC/SQC dùng tâm leave-one-out của nhóm mình",
           "metrics": {"d_r": "Pearson r(x, tâm SQC) − r(x, tâm ADC)", "d_e": "‖x − tâm ADC‖ − ‖x − tâm SQC‖"},
           "use": "mô tả hỗ trợ ánh xạ BAS → LUSC ở chương dữ liệu (QĐ96, V4, Y81); không dùng cho H1 hay chọn cấu hình",
           "elapsed_s": time.time() - t0, "versions": all_versions()}
    write_config(out, prov, cfg)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(summary.round(4).to_string(index=False))
        print(ihc.round(3).to_string(index=False))
    print("ghi", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
