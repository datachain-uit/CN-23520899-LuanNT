"""scripts/make_w1_r_labels.py: nhãn đọc QĐ97e cho bảng so sánh với ptLasso, Trans-GLM (W1) — dữ liệu giả."""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
from rq2_fake_zip import patch_run  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("make_w1_r_labels_under_test", REPO / "scripts" / "make_w1_r_labels.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load()


def _row(comp="ptlasso", n_t="5", pool="gpl570", ivw=(0.001, 0.02), eq=(0.002, 0.03)):
    return {"comparison": comp, "n_t": n_t, "pool_set": pool, "cohorts": "gse10245 gse30219 gse37745", "delta_ivw": sum(ivw) / 2,
            "ci95_ivw_low": ivw[0], "ci95_ivw_high": ivw[1], "delta_eq": sum(eq) / 2, "ci95_eq_low": eq[0], "ci95_eq_high": eq[1]}


@pytest.mark.parametrize("ivw, eq, want", [
    ((0.001, 0.02), (0.002, 0.03), "MVP higher"),
    ((0.001, 0.02), (-0.001, 0.03), "inconclusive"),                 # một cách gộp chứa 0
    ((-0.008, 0.009), (-0.009, 0.005), "equivalent within ±0.01"),
    ((-0.008, 0.011), (-0.009, 0.005), "inconclusive"),              # một CI ra ngoài (−0,01; 0,01)
    ((-0.03, -0.002), (-0.04, -0.001), "baseline higher"),
    ((-0.03, 0.001), (-0.04, -0.001), "inconclusive"),
])
def test_qd97e_rule(ivw, eq, want):
    assert m.label_row(pd.Series(_row(ivw=ivw, eq=eq)))["label"] == want


def test_overlap_of_higher_and_equivalent_follows_rule_order_and_is_flagged():
    r = m.label_row(pd.Series(_row(ivw=(0.001, 0.008), eq=(0.002, 0.009))))
    assert r["label"] == "MVP higher" and r["rule_equivalent"] and r["also_equivalent"]
    r = m.label_row(pd.Series(_row(ivw=(-0.008, -0.001), eq=(-0.009, -0.002))))
    assert r["label"] == "equivalent within ±0.01" and r["rule_baseline_higher"] and not r["also_equivalent"]


def test_undefined_ci_is_inconclusive_and_flagged():
    r = m.label_row(pd.Series(_row(ivw=(np.nan, np.nan), eq=(0.002, 0.003))))
    assert r["label"] == "inconclusive" and r["ci_undefined"]


def _deg():
    rows = []
    for c, rate in (("gse10245", 0.2), ("gse30219", 0.5), ("gse37745", 0.8), ("gse42127", 1.0)):
        for k in (2, 3, 4, 5, 10):
            rows.append({"cohort": c, "n_t": "5", "k": k, "n_runs": 20, "n_defined": 20 if c != "gse37745" else 10,
                         "same_as_stage1_topk_rate": rate if k != 4 else 0.0, "alpha_mode": 0.25})
    return pd.DataFrame(rows)


def test_stage1_rate_sits_next_to_every_ptlasso_row_only():
    base = pd.DataFrame([_row(), _row(comp="transglm"), _row(pool="all4") | {"cohorts": "gse10245 gse30219 gse37745 gse42127"}])
    t = m.label_table(base, _deg())
    gpl, tg, all4 = t.iloc[0], t.iloc[1], t.iloc[2]
    assert gpl.same_as_stage1_topk_rate == pytest.approx((0.2 * 20 + 0.5 * 20 + 0.8 * 10) / 50)   # k = 4 bị loại
    assert gpl.same_as_stage1_topk_min == 0.2 and gpl.same_as_stage1_topk_max == 0.8 and gpl.n_defined == 150
    assert np.isnan(tg.same_as_stage1_topk_rate)
    assert all4.same_as_stage1_topk_max == 1.0
    pd.testing.assert_frame_equal(t[list(base.columns)], base)          # số gốc không đổi


def test_main_writes_labeled_table_caption_and_config(tmp_path, monkeypatch):
    base = pd.DataFrame([_row(), _row(comp="transglm", ivw=(-0.03, -0.002), eq=(-0.04, -0.001))])
    z = tmp_path / "rq2_tables_fake_results.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("results/survey_v2/RQ2_tables/rq2_baselines.csv", base.to_csv(index=False))
        zf.writestr("results/survey_v2/RQ2_tables/rq2_ptlasso_degeneracy.csv", _deg().to_csv(index=False))
        zf.writestr("results/survey_v2/RQ2_tables/config.json", json.dumps({"commit_hash": "2b971e8abc", "results_opened_at_utc": "2026-11-09 07:47:29 UTC",
                                                                           "traceable": True}))
    patch_run(monkeypatch, m, tmp_path / "out")
    assert m.main(["--rq2-tables", str(z)]) == 0
    out = tmp_path / "out" / "RQ2_r_labels"
    t = pd.read_csv(out / "rq2_baselines_labeled.csv", dtype={"n_t": str}, encoding="utf-8-sig")
    assert list(t.label) == ["MVP higher", "baseline higher"]
    cap = (out / "rq2_baselines_caption.txt").read_text(encoding="utf-8")
    assert "2026-11-09 07:47:29 UTC" in cap and "2b971e8" in cap and "số không đổi" in cap and "không Holm" in cap
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["source_table_commit"] == "2b971e8abc" and cfg["labels"] == list(m.LABELS) and cfg["choices_to_confirm"]
