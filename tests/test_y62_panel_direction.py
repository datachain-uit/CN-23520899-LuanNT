"""scripts/y62_panel_direction.py (Review 07/11 Y62; Review 08/11 Mục 8.7, QĐ71a): chiều LUSC − LUAD của gen panel nguồn E7, DSG3 > KRT5."""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load():
    spec = importlib.util.spec_from_file_location("y62_under_test", REPO / "scripts" / "y62_panel_direction.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load()
PANELS = {"2": ["DSG3", "KRT5"], "3": ["DSC3", "DSG3", "KRT5"], "4": ["DSC3", "DSG3", "HNF1B", "KRT5"],
          "5": ["ARHGEF38", "DSC3", "DSG3", "HNF1B", "KRT5"], "10": ["ARHGEF38", "DSC3", "DSG3", "HNF1B", "KRT5", "TMC5"]}


def _frame(seed=0, n0=30, n1=20, flip_pair=False, drop=()):
    """LUSC (1) cao DSG3, KRT5, DSC3; LUAD (0) cao HNF1B. flip_pair: KRT5 cao hơn DSG3 ở LUSC (đảo hiệu DSG3 − KRT5)."""
    rng = np.random.default_rng(seed)
    y = np.array([0] * n0 + [1] * n1)
    X = pd.DataFrame({g: rng.normal(5, 1, len(y)) for g in ["DSG3", "KRT5", "DSC3", "HNF1B", "ARHGEF38", "TMC5", "OTHER"]})
    X["DSG3"] += 4 * y + (0 if flip_pair else 1.0 * y)
    X["KRT5"] += 4 * y + (1.5 * y if flip_pair else 0)
    X["DSC3"] += 3 * y
    X["HNF1B"] -= 2 * y
    X = X.drop(columns=list(drop))
    X.index = [f"s{i}" for i in range(len(y))]
    return X, y


def test_source_panels_are_read_from_e7_config_and_nested_entry_k(tmp_path):
    for c in ("gse10245", "gse30219"):
        d = tmp_path / "E7" / c
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"source_panels": PANELS, "commit_hash": "9b2d6a5abc"}), encoding="utf-8")
    entry, meta = m.panel_entry_k(tmp_path / "E7", ["gse10245", "gse30219"], k_max=5)
    assert entry == {"DSG3": 2, "KRT5": 2, "DSC3": 3, "HNF1B": 4, "ARHGEF38": 5}
    assert meta["commit_hash"] == {"gse10245": "9b2d6a5abc", "gse30219": "9b2d6a5abc"}


def test_panels_must_agree_across_cohorts_and_be_nested(tmp_path):
    for c, panels in (("a", PANELS), ("b", {**PANELS, "3": ["DSC3", "DSG3", "TMC5"]})):
        d = tmp_path / "E7" / c
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"source_panels": panels}), encoding="utf-8")
    with pytest.raises(ValueError, match="khác nhau|lồng"):
        m.panel_entry_k(tmp_path / "E7", ["a", "b"], k_max=5)


def test_panel_entry_k_reads_zip_like_directory(tmp_path):
    z = tmp_path / "e7.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("results/survey_v2/E7/gse37745/config.json", json.dumps({"source_panels": PANELS, "commit_hash": "x"}))
    entry, _ = m.panel_entry_k(z, ["gse37745"], k_max=5)
    assert entry["HNF1B"] == 4


def test_gene_direction_matches_scipy_welch_and_flags_missing_genes():
    X, y = _frame()
    entry = {"DSG3": 2, "KRT5": 2, "DSC3": 3, "HNF1B": 4, "ARHGEF38": 5}
    table = m.gene_direction({"toy": (X.drop(columns="ARHGEF38"), y, "GPL570")}, entry).set_index("gene")
    t = stats.ttest_ind(X.loc[y == 1, "DSG3"], X.loc[y == 0, "DSG3"], equal_var=False)
    assert table.loc["DSG3", "welch_t"] == pytest.approx(t.statistic) and table.loc["DSG3", "p_value"] == pytest.approx(t.pvalue)
    assert table.loc["DSG3", "higher_in"] == "LUSC" and table.loc["HNF1B", "higher_in"] == "LUAD"
    assert table.loc["DSG3", "diff_lusc_minus_luad"] == pytest.approx(X.loc[y == 1, "DSG3"].mean() - X.loc[y == 0, "DSG3"].mean())
    assert bool(table.loc["ARHGEF38", "present"]) is False and np.isnan(table.loc["ARHGEF38", "welch_t"])
    assert (table.loc["DSG3", "n_luad"], table.loc["DSG3", "n_lusc"], table.loc["DSG3", "k_enter"]) == (30, 20, 2)


def test_pair_table_counts_dsg3_gt_krt5_by_class_and_auroc_of_difference():
    X, y = _frame(flip_pair=True)
    rows = m.pair_table({"toy": (X, y, "GPL570")}, "DSG3", "KRT5").set_index("class")
    for cls, lab in (("LUAD", 0), ("LUSC", 1)):
        want = int((X.loc[y == lab, "DSG3"] > X.loc[y == lab, "KRT5"]).sum())
        assert rows.loc[cls, "n_a_gt_b"] == want and rows.loc[cls, "frac_a_gt_b"] == pytest.approx(want / (y == lab).sum())
    assert rows.loc["all", "auroc_a_minus_b"] == pytest.approx(roc_auc_score(y, X["DSG3"] - X["KRT5"]))
    assert rows.loc["all", "auroc_a_minus_b"] < 0.5                 # fixture: LUSC có KRT5 tăng nhiều hơn DSG3, hiệu đảo chiều


def test_build_covers_every_dataset_and_gene():
    frames = {"tcga": _frame(seed=1), "gse_x": _frame(seed=2, flip_pair=True)}
    frames = {k: (X, y, "p") for k, (X, y) in frames.items()}
    entry = {"DSG3": 2, "KRT5": 2, "DSC3": 3, "HNF1B": 4, "ARHGEF38": 5}
    direction, pairs = m.build(frames, entry)
    assert len(direction) == 2 * 5 and set(direction.dataset) == {"tcga", "gse_x"}
    assert set(pairs["class"]) == {"LUAD", "LUSC", "all"} and len(pairs) == 6
    s = direction.pivot(index="gene", columns="dataset", values="higher_in")
    assert (s.loc[["DSG3", "KRT5", "DSC3"]] == "LUSC").all().all() and (s.loc["HNF1B"] == "LUAD").all()
