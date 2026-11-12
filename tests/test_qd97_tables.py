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


# ------------------------------------------------------------------------------------------------- QĐ97a theo từng k (W4)
def test_decomposition_by_k_covers_every_k_and_cohort():
    out = m.decomposition_by_k(_data(), n_boot=20)
    cells, pooled = out["cells"], out["pooled"]
    assert set(cells.k) == {"2", "3", "4", "5", "10"} and set(cells.cohort) == set(COHORTS)
    assert {"k", "pool_set", "i2"} <= set(pooled.columns) and set(pooled.k) == {"2", "3", "4", "5", "10"}


def test_delta_pi_is_zero_when_mvp_equals_to_source():
    chosen = {(s, u, n, k): 0.0 for s in range(2) for u in range(2) for n in ("5", "10", "20", "40") for k in (2, 3, 4, 5, 10)}
    data = {c: synthetic(cohort=c, n_ts=("5", "10", "20", "40"), ks=(2, 3, 4, 5, 10), seed=i, chosen=chosen) for i, c in enumerate(COHORTS)}
    t = m.decomposition_by_k(data, n_boot=10)["cells"]
    assert np.abs(t[t.comparison == "decomp_pi"].delta).max() < 1e-12


def test_space_plus_pi_equals_h2_delta_run_by_run():
    per, sc = _data()["gse37745"]
    for k in ((2,), (2, 3, 5)):
        tot = m.m2.collect_runs(per, sc, "gse37745", "5", "mvp", "target_only", k)
        sp = m.m2.collect_runs(per, sc, "gse37745", "5", m.TO_SOURCE, "target_only", k)
        pi = m.m2.collect_runs(per, sc, "gse37745", "5", "mvp", m.TO_SOURCE, k)
        for a, b, c in zip(tot, sp, pi):
            assert (a.split, a.subset) == (b.split, b.subset) == (c.split, c.subset)
            assert np.allclose(a.D, b.D + c.D, atol=1e-12)


# ------------------------------------------------------------------------------------------------- main() (Y90)
def _run_main(tmp_path, monkeypatch, with_rq4: bool):
    import json
    from rq2_fake_zip import cohort_data, patch_run, write_rq2_zip, write_rq4_zip
    main = cohort_data()
    rq2 = write_rq2_zip(tmp_path / "rq2_mvp_fake_results.zip", main)
    argv = ["--rq2", str(rq2), "--n-boot", "10"]
    if with_rq4:
        argv += ["--rq4", str(write_rq4_zip(tmp_path / "rq4_h4b_noihc_fake_results.zip", main))]
    patch_run(monkeypatch, m, tmp_path / "out")
    assert m.main(argv) == 0
    out = tmp_path / "out" / "RQ2_qd97_tables"
    return {p.stem for p in out.glob("*.csv")}, json.loads((out / "config.json").read_text(encoding="utf-8"))


def test_main_writes_every_table_and_config(tmp_path, monkeypatch):
    names, cfg = _run_main(tmp_path, monkeypatch, with_rq4=True)
    assert names == {"qd97a_decomposition_cells", "qd97a_decomposition_pooled", "qd97a_decomposition_by_k_cells", "qd97a_decomposition_by_k_pooled",
                     "qd97b_true_vs_shuffled_cells", "qd97b_true_vs_shuffled_pooled", "qd97d_cohort_table", "qd97d_secondary_metrics",
                     "qd97d_gse42127_by_k"}
    assert cfg["commit_hash"] == "test000" and cfg["qd97b_skipped"] is None and len(cfg["input_sha256"]) == 2
    assert cfg["results_opened_at_utc"].endswith("UTC")


def test_main_without_rq4_skips_table_b_and_says_so(tmp_path, monkeypatch):
    names, cfg = _run_main(tmp_path, monkeypatch, with_rq4=False)
    assert not any(n.startswith("qd97b_") for n in names) and "qd97a_decomposition_by_k_cells" in names
    assert cfg["rq4_input"] is None and cfg["qd97b_skipped"]
