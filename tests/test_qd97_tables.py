"""scripts/make_qd97_tables.py: bảng bổ sung QĐ97 (Review 09/11) — phân rã, bảng từng cohort, chỉ số phụ, bảng GSE42127, MVP thật − xáo nhãn."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
from test_rq2_tables import synthetic  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("make_qd97_tables_under_test", REPO / "scripts" / "make_qd97_tables.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load()
COHORTS = ("gse10245", "gse30219", "gse37745", "gse42127")


def _data(seed0=10, shift=0.6, ks=(2, 3, 4, 5, 10)):
    return {c: synthetic(cohort=c, n_ts=("5", "10") if c == "gse10245" else ("5", "10", "20", "40"), ks=ks, seed=seed0 + i, mvp_shift=shift)
            for i, c in enumerate(COHORTS)}


def test_decomposition_has_d3_terms_and_adds_up():
    t = m.decomposition(_data(), n_boot=20)["cells"]
    assert {"decomp_total", "decomp_space", "decomp_pi", "decomp_d3_select_on_source", "decomp_d3_pooled"} <= set(t.comparison)
    d3 = t[t.comparison == "decomp_d3_pooled"]
    assert set(d3.n_t) <= {"5", "10", "20"}                       # D3 chỉ có ở n_T ≤ 20
    s = t.set_index(["cohort", "n_t", "comparison"]).delta
    for (c, n_t, comp), v in s.items():
        if comp == "decomp_total":
            assert v == pytest.approx(s[(c, n_t, "decomp_space")] + s[(c, n_t, "decomp_pi")], abs=1e-12)


def test_cohort_table_matches_rq2_build_and_adds_two_sided_ci():
    data = _data()
    t = m.cohort_table(data, n_boot=20)
    ref = m.m2.build(data, n_boot=20)["cells"]
    cells = t[t.level == "cohort"]
    merged = cells.merge(ref, on=["cohort", "comparison", "n_t"], suffixes=("", "_ref"))
    assert len(merged) == len(ref) and np.allclose(merged.delta, merged.delta_ref) and np.allclose(merged.se, merged.se_ref)
    assert np.allclose(cells.ci95_low, cells.delta - 1.959963984540054 * cells.se)
    pooled = t[t.level == "pooled"]
    assert {"i2", "w_gse30219"} <= set(pooled.columns) and set(pooled.pool_set) == {"gpl570", "all4"}


def test_secondary_metrics_pair_mvp_with_each_arm():
    data = _data()
    per, sc = data["gse30219"]
    per = per.copy()
    per.loc[(per.gamma_rule == "m5_pos"), "bacc"] = 0.8               # MVP: bacc 0,8; còn lại 0,5
    data["gse30219"] = (per, sc)
    t = m.secondary_metrics(data)
    r = t[(t.cohort == "gse30219") & (t.n_t == "5") & (t.metric == "bacc") & (t.arm == "target_only")].iloc[0]
    assert r.mvp_mean == pytest.approx(0.8) and r.arm_mean == pytest.approx(0.5) and r.delta_mean == pytest.approx(0.3)
    assert {"select_on_source", "pooled", m.TO_SOURCE} <= set(t.arm) and {"bacc", "mcc", "logloss", "brier"} <= set(t.metric)
    pooled = t[t.level == "pooled"]
    assert {"gpl570", "all4"} == set(pooled.cohort)               # trung bình bằng nhau các cohort
    g = t[(t.level == "cohort") & (t.metric == "bacc") & (t.arm == "target_only") & (t.n_t == "5") & t.cohort.isin(["gse30219", "gse37745", "gse10245"])]
    p = pooled[(pooled.cohort == "gpl570") & (pooled.metric == "bacc") & (pooled.arm == "target_only") & (pooled.n_t == "5")].iloc[0]
    assert p.delta_mean == pytest.approx(g.delta_mean.mean())


def test_gse42127_table_reports_each_k_and_k_avg():
    t = m.cohort_by_k(_data(), n_boot=20, cohort="gse42127")
    assert set(t.k) == {"2", "3", "4", "5", "10", "2 3 5"}
    assert {"target_only", "select_on_source", "pooled", "decomp_space", "decomp_pi"} <= set(t.comparison)
    assert (t.cohort == "gse42127").all()


def test_true_minus_shuffled_pairs_runs():
    data = _data()
    shuf = _data(seed0=50, shift=0.0)
    t = m.true_vs_shuffled(data, shuf, n_boot=20)["cells"]
    row = t[(t.cohort == "gse37745") & (t.n_t == "5")].iloc[0]
    sel = lambda f: f[(f.gamma_rule == "m5_pos") & (f.n_t == "5") & f.k.isin([2, 3, 5])].set_index(["split", "subset", "k"]).auroc  # noqa: E731
    a, b = data["gse37745"][0], shuf["gse37745"][0]
    assert row.delta == pytest.approx((sel(a) - sel(b)).groupby(level=["split", "subset"]).mean().mean(), abs=1e-12)
