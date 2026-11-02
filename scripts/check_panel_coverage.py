"""Độ phủ gen của các panel cố định (IHC-2, IHC-4, Girard 2016, Wilkerson 2013) trên từng bộ dữ liệu (G-14, R1-04, Review 31/10 dòng 267).

Hai không gian gen cho mỗi bộ:
* native     — mọi cột gen của expression.h5 (nền tảng gốc);
* e1         — không gian E1/E1b thực dùng: load_dataset(gene_space="reference") = giao với gen TCGA (TCGA giữ nguyên).
Đầu ra: <out>/panel_gene_coverage.csv (một dòng mỗi panel × bộ × không gian gen).

    python scripts/check_panel_coverage.py --data <KLTN_DATA_ROOT> --panels survey_v2/panels/published_panels.csv --out <KLTN_DATA_ROOT>/qc
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

COHORTS = {  # tên -> (đường dẫn tương đối, nền tảng)
    "tcga_luad_lusc_primary_tumor": ("experiment-data-perm/tcga_luad_lusc_primary_tumor", "TCGA RNA-seq"),
    "gse10245": ("experiment-data-geo-target/gse10245", "GPL570"),
    "gse30219": ("experiment-data-geo-target/gse30219", "GPL570"),
    "gse37745": ("experiment-data-geo-target/gse37745", "GPL570"),
    "gse42127": ("experiment-data-geo-target/gse42127", "GPL6884"),
}
REFERENCE = "tcga_luad_lusc_primary_tumor"
IHC = {"ihc_2": ["NKX2-1", "TP63"], "ihc_4": ["NKX2-1", "TP63", "NAPSA", "KRT5"]}  # survey_v2/config.py IHC_2, IHC_4


def genes_of(path: Path) -> list[str]:
    return [str(g) for g in pd.read_hdf(path).columns]  # cùng cách đọc với survey_v2/data.py


def published(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(line for line in f if not line.startswith("#")):
            if r.get("gene"):
                out.setdefault(f"panel_{r['panel'].strip()}", []).append(r["gene"].strip())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--panels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    panels = {**IHC, **published(a.panels)}
    native = {c: genes_of(a.data / rel / "expression.h5") for c, (rel, _) in COHORTS.items()}
    ref = set(native[REFERENCE])
    rows = []
    for c, (_, platform) in COHORTS.items():
        spaces = {"native": set(native[c]), "e1": set(native[c]) if c == REFERENCE else set(native[c]) & ref}
        for space, genes in spaces.items():
            for p, members in panels.items():
                miss = [g for g in members if g not in genes]
                rows.append({"panel": p, "cohort": c, "platform": platform, "gene_space": space, "n_genes_space": len(genes),
                             "n_panel": len(members), "n_present": len(members) - len(miss), "missing": ";".join(miss),
                             "complete": not miss})
    a.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(a.out / "panel_gene_coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["checked_at_utc"])
        w.writeheader(); w.writerows([{**r, "checked_at_utc": stamp} for r in rows])
    for r in rows:
        if r["gene_space"] == "e1":
            print(f"{r['panel']:<22} {r['cohort']:<30} {r['n_present']}/{r['n_panel']} thiếu: {r['missing'] or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
