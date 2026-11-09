"""Biến thể gse30219_bas (độ nhạy RQ1 phần (b), QĐ53b; Review 08/11 Mục 8.5 bước 1–2; QĐ67 N4): BAS → LUSC.

Dựng từ series matrix GSE30219 với ĐÚNG định nghĩa gen của bản R5: mỗi gen của gse30219/expression.h5 (141 mẫu sau QC) được ghép với
probe duy nhất của series matrix có cùng giá trị float32 trên 141 mẫu đó (gen không khớp hoặc khớp nhiều probe -> dừng). Như vậy 141
mẫu R5 giữ nguyên từng byte; chỉ thêm mẫu histology = BAS (basaloid squamous cell carcinoma, thuộc nhóm carcinoma tế bào vảy theo WHO
2015, Travis2015) với nhãn LUSC = 1. Không chạy lại collapse_probes (chọn probe theo phương sai trên tập mẫu mới sẽ đổi gen).

QC trên TOÀN BỘ mẫu của biến thể (5 mẫu R5 đã loại + 141 + BAS):
  * kế thừa mọi mẫu trong gse30219/excluded_samples_gse30219.csv;
  * giới tính (QĐ22, scripts/check_sex_consistency.py::sex_table, k-means trên chính biến thể): cách ly verdict opposite / inconsistent;
  * mẫu trùng (QĐ21, như scripts/geo_sample_qc.py rz-screen): cặp r_z ≥ 0,90 hoặc r thô ≥ 0,99 -> cách ly cả hai mẫu.
Mẫu R5 bị cờ mới (không có trong danh sách cũ) cũng cách ly theo cùng quy tắc và ghi rõ trong config (r5_newly_flagged).

    python scripts/build_gse30219_bas.py            # $KLTN_DATA_ROOT/experiment-data-geo-target/gse30219_bas
    python scripts/build_e1_repeated_splits.py --datasets gse30219_bas      # bước 3: bản chia 5×5 mới, seed 42–46

Đầu ra: <out>/{expression.h5, metadata.csv, excluded_samples_gse30219_bas.csv, label_mapping.json, config.json,
qc/sex_check_gse30219_bas.csv, qc/rz_screen_pairs_gse30219_bas.csv}. Không ghi đè thư mục đã có.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import sha256  # noqa: E402
from survey_v2.runlog import DATA_ROOT  # noqa: E402
from scripts.check_sex_consistency import sex_table  # noqa: E402
from scripts.geo_sample_qc import parse_sex, row_corr, zscore_genes  # noqa: E402
from scripts.provenance import require, stamp  # noqa: E402

NAME = "gse30219_bas"
BASE = "gse30219"
LABELS = {"ADC": ("LUAD", 0), "SQC": ("LUSC", 1), "BAS": ("LUSC", 1)}
RZ_THRESHOLD, RAW_THRESHOLD = 0.90, 0.99            # như geo_sample_qc.py rz-screen (QĐ21)
QUARANTINE = ("opposite", "inconsistent")           # QĐ22
RAW = DATA_ROOT / "geo_target_work" / "raw"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series-matrix", type=Path, default=RAW / "GSE30219" / "GSE30219_series_matrix.txt.gz")
    ap.add_argument("--annotation", type=Path, default=RAW / "GPL570" / "GPL570.annot.gz")
    ap.add_argument("--base-dir", type=Path, default=None, help="thư mục gse30219 (bản R5 sau QC); mặc định theo registry")
    ap.add_argument("--out", type=Path, default=None, help=f"mặc định $KLTN_DATA_ROOT/{C.GEO_TARGET_DIR}/{NAME}")
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------------------------- đọc GEO (như notebook dựng cohort)
def parse_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(biểu hiện probe × mẫu, metadata mẫu) — cùng cách đọc với notebooks/phase_0/00_data/geo_luad_lusc_target_cohorts_5fold.ipynb."""
    sample_rows, values, inside = {}, [], False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line.startswith("!series_matrix_table_begin"):
                inside = True
                continue
            if line.startswith("!series_matrix_table_end"):
                inside = False
                continue
            if inside:
                values.append(line)
            elif line.startswith("!Sample_"):
                key, *row = line.split("\t")
                sample_rows.setdefault(key[1:], []).append([x.strip().strip('"') for x in row])
    flat = {}
    for key, rows in sample_rows.items():
        for occurrence, row in enumerate(rows, start=1):
            flat[key if len(rows) == 1 else f"{key}_{occurrence}"] = row
    meta = pd.DataFrame(flat).set_index("Sample_geo_accession")
    meta.index.name = "sample_id"
    expr = pd.read_csv(io.StringIO("\n".join(values)), sep="\t", index_col=0, low_memory=False)
    expr = expr.loc[:, expr.columns.intersection(meta.index)]
    return expr.apply(pd.to_numeric, errors="coerce"), meta


def load_symbols(path: Path) -> pd.Series:
    """probe -> gene symbol đầu tiên (như load_platform_annotation của notebook dựng cohort)."""
    rows, inside = [], False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!platform_table_begin"):
                inside = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if inside:
                rows.append(line)
    ann = pd.read_csv(io.StringIO("".join(rows)), sep="\t", dtype=str)
    id_col = next(c for c in ("ID", "ID_REF", "Probe Set ID") if c in ann)
    gene_col = next(c for c in ("Gene Symbol", "Gene symbol", "GENE_SYMBOL", "Symbol") if c in ann)
    sym = ann.set_index(id_col)[gene_col].fillna("").str.split(" /// ").str[0].str.strip()
    return sym[(sym != "") & ~sym.str.contains("///", regex=False)]


def probe_map(expr: pd.DataFrame, ref: pd.DataFrame, symbols: pd.Series) -> dict:
    """gene -> probe duy nhất có cùng giá trị float32 với bản tham chiếu trên mọi mẫu của nó."""
    E = expr.loc[:, ref.index].astype(np.float32)
    sym = symbols.reindex(E.index).dropna()
    groups = sym.groupby(sym).groups
    out, bad = {}, []
    for g in ref.columns:
        col = ref[g].to_numpy(np.float32)
        hits = [p for p in groups.get(g, []) if np.array_equal(E.loc[p].to_numpy(np.float32), col)]
        if len(hits) != 1:
            bad.append((g, len(hits)))
        else:
            out[g] = hits[0]
    if bad:
        raise ValueError(f"{len(bad)} gen không khớp đúng một probe của series matrix (vd. {bad[:5]}): bản tham chiếu không tái lập được")
    return out


def histology(meta: pd.DataFrame) -> pd.Series:
    cols = [c for c in meta.columns if c.startswith("Sample_characteristics")]
    found = meta[cols].apply(lambda c: c.str.extract(r"^histology:\s*(.*)$", expand=False)).bfill(axis=1).iloc[:, 0]
    return found.str.strip()


@dataclass
class Variant:
    X: pd.DataFrame            # mẫu × gen, float32, mọi mẫu ADC/SQC/BAS (kể cả mẫu R5 đã loại)
    meta: pd.DataFrame
    r5_excluded: pd.DataFrame
    probes: dict
    sources: dict


def build_variant(series_matrix: Path, annotation: Path, base_dir: Path) -> Variant:
    expr, raw_meta = parse_series_matrix(series_matrix)
    ref = pd.read_hdf(base_dir / "expression.h5", key="expression")
    ref.index = ref.index.astype(str)
    ref_meta = pd.read_csv(base_dir / "metadata.csv", index_col="sample_id")
    r5_excluded = pd.read_csv(base_dir / f"excluded_samples_{BASE}.csv", dtype=str)
    pmap = probe_map(expr, ref, load_symbols(annotation))
    hist = histology(raw_meta)
    keep = hist[hist.isin(LABELS)]
    bas = [s for s in keep.index if keep[s] == "BAS"]
    order = list(ref.index) + [s for s in r5_excluded.sample_id if s in keep.index] + bas
    missing = [s for s in order if s not in keep.index]
    if missing or len(order) != len(keep) or len(set(order)) != len(order):
        raise ValueError(f"tập mẫu R5 + BAS không khớp series matrix (thiếu {missing[:5]}, {len(order)} so với {len(keep)} mẫu ADC/SQC/BAS)")
    genes = list(ref.columns)
    X = pd.DataFrame(expr.loc[[pmap[g] for g in genes], order].to_numpy(np.float32).T, index=pd.Index(order, name="sample_id"), columns=genes)
    if not np.array_equal(X.loc[ref.index].to_numpy(np.float32), ref.to_numpy(np.float32)):
        raise ValueError("biểu hiện của các mẫu R5 dựng lại khác bản tham chiếu")
    meta = raw_meta.loc[order].copy()
    meta["histology_raw"] = hist.loc[order]
    meta["class"] = meta.histology_raw.map(lambda h: LABELS[h][0])
    meta["label"] = meta.histology_raw.map(lambda h: LABELS[h][1]).astype(int)
    meta["patient_id"] = meta.index                                   # GSE30219: một mẫu mỗi bệnh nhân (data card 02/11)
    meta["sex"] = parse_sex(meta.reset_index())[0].set_axis(meta.index)
    meta["variant_group"] = np.where(meta.histology_raw == "BAS", "bas", "r5")
    if "label" in ref_meta and not (ref_meta.label.reindex(ref.index).to_numpy() == meta.loc[ref.index, "label"].to_numpy()).all():
        raise ValueError("nhãn của mẫu R5 khác bản tham chiếu")
    if "sex" in ref_meta:
        old = ref_meta.sex.reindex(ref.index).fillna("").astype(str).to_numpy()
        if not (old == meta.loc[ref.index, "sex"].astype(str).to_numpy()).all():
            raise ValueError("giới tính của mẫu R5 khác bản tham chiếu")
    sources = {"series_matrix": str(series_matrix), "series_matrix_sha256": sha256(series_matrix), "annotation": str(annotation),
               "annotation_sha256": sha256(annotation), "base_expression_sha256": sha256(base_dir / "expression.h5"),
               "base_metadata_sha256": sha256(base_dir / "metadata.csv")}
    return Variant(X, meta, r5_excluded, pmap, sources)


# ------------------------------------------------------------------------------------------------- QC
def run_qc(v: Variant) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(mẫu cách ly kèm lý do, bảng giới tính, cặp trùng). Trên toàn bộ mẫu của biến thể."""
    reasons: dict[str, list] = {}
    for r in v.r5_excluded.itertuples(index=False):
        reasons.setdefault(r.sample_id, []).append(f"kế thừa R5: {r.reason}")
    ds = SimpleNamespace(genes=np.asarray(v.X.columns), X=v.X.to_numpy(float), sample_ids=np.asarray(v.X.index), y=v.meta.label.to_numpy(int))
    sex, info = sex_table(ds, v.meta[["sex"]])
    for r in sex.itertuples(index=False):
        if r.verdict in QUARANTINE:
            reasons.setdefault(r.sample_id, []).append(f"QĐ22 sex check: {r.verdict} (meta_sex = {r.meta_sex}, biểu hiện = {r.category})")
    Xv = v.X.to_numpy(float)
    Rz, Rr = row_corr(zscore_genes(Xv), zscore_genes(Xv)), row_corr(Xv, Xv)
    iu = np.triu_indices(len(Xv), 1)
    hit = (Rz[iu] >= RZ_THRESHOLD) | (Rr[iu] >= RAW_THRESHOLD)
    ids = v.X.index
    pairs = pd.DataFrame([{"sample_a": ids[i], "sample_b": ids[j], "r_z": round(float(Rz[i, j]), 4), "r_raw": round(float(Rr[i, j]), 4),
                           "group_a": v.meta.variant_group.iloc[i], "group_b": v.meta.variant_group.iloc[j]}
                          for i, j in zip(iu[0][hit], iu[1][hit])], columns=["sample_a", "sample_b", "r_z", "r_raw", "group_a", "group_b"])
    pairs = pairs.sort_values("r_z", ascending=False, ignore_index=True)
    for p in pairs.itertuples(index=False):
        for s, other in ((p.sample_a, p.sample_b), (p.sample_b, p.sample_a)):
            reasons.setdefault(s, []).append(f"QĐ21 mẫu trùng với {other}: r_z = {p.r_z}, r thô = {p.r_raw}")
    excluded = pd.DataFrame([{"sample_id": s, "reason": "; ".join(r), "variant_group": v.meta.variant_group.get(s, "")} for s, r in reasons.items()])
    sex.attrs["info"] = info
    return excluded, sex, pairs


# ------------------------------------------------------------------------------------------------- ghi
def write_variant(v: Variant, out: Path, prov: dict) -> dict:
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} đã có: không ghi đè dữ liệu (xoá hoặc đổi tên sau khi kiểm)")
    excluded, sex, pairs = run_qc(v)
    drop = set(excluded.sample_id)
    kept = [s for s in v.X.index if s not in drop]
    X, meta = v.X.loc[kept], v.meta.loc[kept]
    out.mkdir(parents=True)
    (out / "qc").mkdir()
    X.to_hdf(out / "expression.h5", key="expression", mode="w", complevel=9, complib="blosc")
    meta.to_csv(out / "metadata.csv", index_label="sample_id")
    excluded.to_csv(out / f"excluded_samples_{NAME}.csv", index=False)
    (out / "label_mapping.json").write_text(json.dumps({"LUAD": 0, "LUSC": 1, "histology": {k: v_[0] for k, v_ in LABELS.items()}}, indent=2), encoding="utf-8")
    sex.to_csv(out / "qc" / f"sex_check_{NAME}.csv", index=False)
    pairs.to_csv(out / "qc" / f"rz_screen_pairs_{NAME}.csv", index=False)
    r5_old = set(v.r5_excluded.sample_id)
    n_lusc = int((meta.label == 1).sum())
    n_bas = int((meta.histology_raw == "BAS").sum())
    cfg = {"dataset": NAME, "base_dataset": BASE, "geo_accession": "GSE30219", "task": "LUAD vs LUSC; BAS (basaloid) gộp vào LUSC (WHO 2015, Travis2015)",
           "label_mapping": {k: v_[0] for k, v_ in LABELS.items()},
           "gene_definition": "probe của bản R5, khôi phục bằng so khớp float32 trên các mẫu R5 (không chọn lại probe theo phương sai)",
           "n_genes": int(X.shape[1]), "n_before_qc": int(len(v.X)), "n_after_qc": int(len(X)),
           "n_per_class": [int((meta.label == 0).sum()), n_lusc], "histology_counts_after_qc": meta.histology_raw.value_counts().to_dict(),
           "histology_counts_before_qc": v.meta.histology_raw.value_counts().to_dict(),
           "n_bas_before_qc": int((v.meta.histology_raw == "BAS").sum()), "n_bas_kept": n_bas, "bas_fraction_of_lusc": n_bas / n_lusc if n_lusc else None,
           "excluded_inherited_r5": sorted(r5_old), "excluded_bas": sorted(s for s in drop if v.meta.variant_group.get(s) == "bas"),
           "r5_newly_flagged": sorted(s for s in drop if v.meta.variant_group.get(s) == "r5" and s not in r5_old),
           "r5_values_identical": True, "n_duplicate_pairs": int(len(pairs)),
           "duplicate_pairs_bas_r5": int(((pairs.group_a != pairs.group_b)).sum()) if len(pairs) else 0,
           "qc_rules": {"sex": "QĐ22: sex_table (k-means trên biến thể), cách ly opposite / inconsistent",
                        "duplicates": f"QĐ21: r_z >= {RZ_THRESHOLD} hoặc r thô >= {RAW_THRESHOLD} trên mọi cặp của biến thể; cách ly cả hai mẫu",
                        "inherited": f"mọi mẫu trong {BASE}/excluded_samples_{BASE}.csv"},
           "sex_qc_info": {k: (list(x) if isinstance(x, tuple) else x) for k, x in sex.attrs.get("info", {}).items()},
           "sources": v.sources,
           "files_sha256": {f: sha256(out / f) for f in ("expression.h5", "metadata.csv", f"excluded_samples_{NAME}.csv")},
           "split": f"chưa có: chạy scripts/build_e1_repeated_splits.py --datasets {NAME} (5×5, seed 42–46)"}
    cfg.update({k: prov[k] for k in prov if k in ("traceable", "commit_hash")})
    if prov and "script" in prov:
        stamp(cfg, prov)
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return cfg


def main(argv=None, script=__file__) -> int:
    from survey_v2.data import resolve_dataset_dir
    args = parse_args(argv)
    base = args.base_dir or resolve_dataset_dir(DATA_ROOT, BASE)
    out = args.out or DATA_ROOT / C.GEO_TARGET_DIR / NAME
    prov = require(script, data_files=[args.series_matrix, args.annotation, base / "expression.h5", base / "metadata.csv"])
    cfg = write_variant(build_variant(args.series_matrix, args.annotation, base), out, prov)
    print(json.dumps({k: cfg[k] for k in ("n_before_qc", "n_after_qc", "n_per_class", "n_bas_before_qc", "n_bas_kept", "bas_fraction_of_lusc",
                                         "excluded_bas", "r5_newly_flagged", "n_duplicate_pairs")}, ensure_ascii=False, indent=2))
    print("ghi", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
