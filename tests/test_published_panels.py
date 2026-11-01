"""Kiểm tra survey_v2/panels/published_panels.csv (Y28, G-14, R1-04).

* Định dạng: cột panel, gene, source; source có DOI; không trùng gen trong một panel;
  không ghi đè IHC built-in; tên method = panel_<panel>.
* Đọc panel từ CSV tạm và ánh xạ gen thiếu theo nền tảng (panel_indices).
* Khi CSV đã điền: đủ hai panel kỳ vọng (Girard 2016, Wilkerson 2013). Khi chưa điền: skip có lý do.
"""

from __future__ import annotations

import csv
import re

import numpy as np
import pytest

from survey_v2.panels import PANELS_CSV, all_fixed_panels, panel_indices, published_panels

EXPECTED_PANELS = {"panel_girard2016", "panel_wilkerson2013"}
DOI_RE = re.compile(r"10\.\d{4,9}/\S+")
SYMBOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-\.]*$")


def _rows(path=PANELS_CSV):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(line for line in f if not line.startswith("#")) if r.get("gene")]


def test_csv_has_expected_header():
    with open(PANELS_CSV, encoding="utf-8") as f:
        header = next(line for line in f if not line.startswith("#"))
    assert [c.strip() for c in header.split(",")] == ["panel", "gene", "source"]


def test_rows_are_well_formed():
    seen: dict[str, set[str]] = {}
    for r in _rows():
        panel, gene, source = r["panel"].strip(), r["gene"].strip(), (r.get("source") or "").strip()
        assert panel not in ("ihc_2", "ihc_4"), "IHC là built-in, không ghi vào CSV"
        assert SYMBOL_RE.match(gene), f"ký hiệu gen không hợp lệ: {gene!r}"
        assert DOI_RE.search(source), f"{panel}/{gene}: cột source phải có DOI"
        assert gene not in seen.setdefault(panel, set()), f"gen trùng trong {panel}: {gene}"
        seen[panel].add(gene)


def test_published_panels_are_filled():
    panels = published_panels()
    if not panels:
        pytest.skip("published_panels.csv chưa điền gen (Y28) — điền từ bảng/bổ sung của bài gốc")
    assert EXPECTED_PANELS <= set(panels), f"thiếu panel: {EXPECTED_PANELS - set(panels)}"
    for name, genes in panels.items():
        assert len(genes) >= 2, f"{name} quá ít gen"


def test_reader_and_platform_mapping(tmp_path):
    p = tmp_path / "panels.csv"
    p.write_text(
        "# chú thích\npanel,gene,source\n"
        "demo,AAA,doi:10.1000/x\n"
        "demo,BBB,doi:10.1000/x\n"
        "ihc_2,ZZZ,doi:10.1000/x\n",
        encoding="utf-8",
    )
    panels = published_panels(p)
    assert panels == {"panel_demo": ["AAA", "BBB"]}          # IHC trong CSV bị bỏ qua
    fixed = all_fixed_panels(p)
    assert fixed["ihc_2"] == ["NKX2-1", "TP63"] and "panel_demo" in fixed
    idx, missing = panel_indices(np.array(["BBB", "CCC", "AAA"]), panels["panel_demo"])
    assert idx.tolist() == [2, 0] and missing == []
    idx, missing = panel_indices(np.array(["AAA"]), panels["panel_demo"])
    assert idx.tolist() == [0] and missing == ["BBB"]
