"""scripts/make_rq4_tables.py: H4a (Nogueira), H4b (nguồn xáo nhãn), ablation RQ2 (phân rã, γ, loại IHC), phân bố γ — dữ liệu giả."""

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
    spec = importlib.util.spec_from_file_location("make_rq4_tables_under_test", REPO / "scripts" / "make_rq4_tables.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load()
COHORTS = ("gse10245", "gse30219", "gse37745", "gse42127")


def _data(seed0=10, shift=0.6, ks=(2, 3, 4, 5)):
    return {c: synthetic(cohort=c, n_ts=("5", "10") if c == "gse10245" else ("5", "10", "20"), ks=ks, seed=seed0 + i, mvp_shift=shift, arms=())
            for i, c in enumerate(COHORTS)}


def _genes(per: pd.DataFrame, n_genes=40, seed=0, stable_mvp=True):
    """Gen của mọi dòng fixed: MVP (γ > 0) gần như cùng một panel; target-only ngẫu nhiên → MVP ổn định hơn."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in per[per.gamma_rule == "fixed"].itertuples():
        if r.gamma > 0 and stable_mvp:
            genes = [f"G{j}" for j in range(r.k)]
        else:
            genes = [f"G{j}" for j in rng.choice(n_genes, size=r.k, replace=False)]
        rows += [{"cohort": r.cohort, "split": r.split, "subset": r.subset, "n_t": r.n_t, "k": r.k, "method": "mvp", "gamma": r.gamma,
                  "prefilter_space": r.prefilter_space, "rank": i + 1, "gene": g} for i, g in enumerate(genes)]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- phân rã, ablation
def test_ablation_no_longer_holds_decomposition_rows_y89():
    """Y89: phân rã chính thức là qd97a (make_qd97_tables.py); bảng ablation không còn decomp_*."""
    cells = m.ablation_tables(_data(), n_boot=10)["cells"]
    assert not any(c.startswith("decomp_") for c in cells.comparison.unique())


def test_ablation_includes_gamma_rules_and_fixed_gammas_against_m5_pos():
    cells = m.ablation_tables(_data(), n_boot=30)["cells"]
    assert {"m5_cv_vs_m5_pos", "gamma_0.5_vs_m5_pos", "gamma_1_vs_m5_pos", "gamma_2_vs_m5_pos"} <= set(cells.comparison)
    pooled = m.ablation_tables(_data(), n_boot=30)["pooled"]
    assert {"ci95_ivw_low", "ci95_eq_high"} <= set(pooled.columns) and (pooled.family == m.DESCRIPTIVE).all()


def test_no_ihc_drop_pairs_variant_runs_with_main_runs():
    data = _data()
    noihc = _data(seed0=40, shift=0.0)                          # biến thể: MVP mất lợi thế
    t = m.noihc_tables(data, noihc, n_boot=30)["cells"]
    row = t[(t.cohort == "gse30219") & (t.n_t == "5") & (t.comparison == "noihc_mvp")].iloc[0]
    per_a, per_b = noihc["gse30219"][0], data["gse30219"][0]
    sel = lambda f: f[(f.gamma_rule == "m5_pos") & (f.n_t == "5") & f.k.isin([2, 3, 5])].set_index(["split", "subset", "k"]).auroc  # noqa: E731
    assert row.delta == pytest.approx((sel(per_a) - sel(per_b)).groupby(level=["split", "subset"]).mean().mean(), abs=1e-12)
    assert set(t.comparison) == {"noihc_mvp", "noihc_target_only"}


# ------------------------------------------------------------------------------------------------- H4b
def test_h4b_uses_margin_and_reports_only():
    shuf = _data(seed0=70, shift=0.0)
    out = m.h4b_tables(shuf, n_boot=30)
    p = out["pooled"]
    assert set(p.n_t) >= {"5", "10", "20"} and (p.margin == 0.01).all() and p.p_holm.isna().all()
    for r in p.itertuples():
        assert r.noninferior == ((r.lower_ivw > -0.01) and (r.lower_eq > -0.01))
        assert r.note == ("" if r.noninferior else m.M3_NOTE)


# ------------------------------------------------------------------------------------------------- H4a
def test_nogueira_panels_follow_m5_pos_gamma_and_stable_mvp_beats_random_target_only():
    data = _data(ks=(2, 3, 5))
    genes = {c: _genes(data[c][0], seed=i) for i, c in enumerate(COHORTS)}
    space = {c: 40 for c in COHORTS}
    cells = m.h4a_cells(data, genes, space)
    r = cells[(cells.cohort == "gse30219") & (cells.n_t == "5")].iloc[0]
    assert r.phi_mvp == pytest.approx(1.0) and r.phi_target_only < 0.6 and r.delta > 0.3 and r.se > 0
    assert r.n_runs == 4


def test_h4a_rule_requires_delta_at_least_0_1_and_ci_excluding_zero():
    def pooled(rows):
        return pd.DataFrame([{"n_t": n, "pool_set": "gpl570", "delta_ivw": a, "ci95_ivw_low": lo, "delta_eq": e, "ci95_eq_low": le}
                             for n, a, lo, e, le in rows])
    ok = pooled([("5", 0.2, 0.05, 0.15, 0.02), ("10", 0.12, 0.01, 0.11, 0.001), ("20", 0.1, 0.0001, 0.1, 0.0001)])
    assert m.decide_h4a(ok)[0] == "ung_ho"
    part = pooled([("5", 0.2, 0.05, 0.15, 0.02), ("10", 0.05, 0.01, 0.11, 0.001), ("20", 0.1, -0.01, 0.1, 0.0001)])
    assert m.decide_h4a(part)[0] == "ung_ho_mot_phan"
    assert m.decide_h4a(pooled([("5", 0.05, 0.01, 0.05, 0.01), ("10", 0.0, -0.1, 0, -0.1), ("20", 0.2, -0.01, 0.2, 0.01)]))[0] == "khong_ung_ho"


# ------------------------------------------------------------------------------------------------- phân bố γ
def test_gamma_distribution_counts_chosen_gamma_by_rule():
    data = _data()
    chosen = {(0, 0, "5", 2): 2.0}
    data["gse30219"] = synthetic(cohort="gse30219", n_ts=("5", "10"), ks=(2, 3, 4, 5), seed=11, chosen=chosen, arms=())
    d = m.gamma_distribution(data)
    row = d[(d.cohort == "gse30219") & (d.rule == "m5_pos") & (d.n_t == "5") & (d.k == 2)].set_index("gamma").n_runs
    assert row.to_dict() == {0.5: 3, 2.0: 1}
    assert set(d.rule) == {"m5_pos", "m5_cv"}


# ------------------------------------------------------------------------------------------------- H4a đủ nhánh (QĐ97c, W2)
def _genes_d3(per: pd.DataFrame, seed=0):
    """Thêm gen cho pooled (ngẫu nhiên) và select-on-source (cố định) vào gen của các dòng fixed."""
    rng = np.random.default_rng(seed)
    rows = [_genes(per, seed=seed)]
    for r in per[per.method.isin(["pooled", "select_on_source"])].itertuples():
        genes = [f"G{j}" for j in (range(r.k) if r.method == "select_on_source" else rng.choice(40, size=r.k, replace=False))]
        rows.append(pd.DataFrame([{"cohort": r.cohort, "split": r.split, "subset": r.subset, "n_t": r.n_t, "k": r.k, "method": r.method,
                                   "gamma": np.nan, "prefilter_space": r.prefilter_space, "rank": i + 1, "gene": g} for i, g in enumerate(genes)]))
    return pd.concat(rows, ignore_index=True)


def test_h4a_reports_each_arm_with_ci_and_pooled_comparison_only_where_d3_ran():
    data = {c: synthetic(cohort=c, n_ts=("5", "10", "40"), ks=(2, 3, 5), seed=i) for i, c in enumerate(COHORTS)}
    genes = {c: _genes_d3(data[c][0], seed=i) for i, c in enumerate(COHORTS)}
    cells = m.h4a_cells(data, genes, {c: 40 for c in COHORTS})
    r = cells[(cells.cohort == "gse30219") & (cells.n_t == "5")].iloc[0]
    for arm in ("mvp", "target_only", "pooled"):
        assert r[f"ci_low_{arm}"] < r[f"phi_{arm}"] < r[f"ci_high_{arm}"] or r[f"se_{arm}"] == 0
    assert r.phi_select_on_source == pytest.approx(1.0) and r.delta_vs_pooled == pytest.approx(r.phi_mvp - r.phi_pooled)
    assert np.isnan(cells[(cells.n_t == "40")].phi_pooled).all()          # D3 không chạy ở n_T = 40
    pooled = m.h4a_pooled(cells)
    assert set(pooled.comparison) == {"mvp_vs_target_only", "mvp_vs_pooled"}
    assert set(pooled[pooled.comparison == "mvp_vs_pooled"].n_t) == {"5", "10"}
    out, levels = m.decide_h4a(pooled)                                    # quyết định chỉ dùng mvp_vs_target_only
    assert [n for n, _ in levels] == ["5", "10"]


# ------------------------------------------------------------------------------------------------- W2 (QĐ125): TO_nguồn, select-on-source, bootstrap theo split
def test_phi_compact_equals_nogueira_on_full_gene_space():
    rng = np.random.default_rng(3)
    d = 40
    panels = [sorted(rng.choice(d, size=3, replace=False)) for _ in range(12)]
    full = np.zeros((12, d))
    for i, p in enumerate(panels):
        full[i, p] = 1
    used = sorted(set().union(*map(set, panels)))
    assert m.phi_compact(full[:, used], d) == pytest.approx(m.nogueira(full)["stability"], abs=1e-12)


def test_h4a_has_to_source_arm_and_select_on_source_without_ci():
    from rq2_fake_zip import cohort_data, genes_for
    data = cohort_data()
    genes = {c: genes_for(data[c][0], seed=i) for i, c in enumerate(COHORTS)}
    cells = m.h4a_cells(data, genes, {c: 40 for c in COHORTS})
    r = cells[(cells.cohort == "gse30219") & (cells.n_t == "5")].iloc[0]
    assert {"phi_to_source", "se_to_source", "ci_low_to_source"} <= set(cells.columns)
    assert r.phi_to_source < 0.6 and r.se_to_source > 0                       # γ = 0 / source: gen ngẫu nhiên trong dữ liệu giả
    assert r.phi_select_on_source == pytest.approx(1.0) and np.isnan(r.se_select_on_source) and np.isnan(r.ci_low_select_on_source)
    assert bool(r.sos_panel_constant)


def test_to_source_panels_come_from_gamma_zero_source_rows():
    from rq2_fake_zip import cohort_data, genes_for
    per = cohort_data()["gse30219"][0]
    genes = genes_for(per, seed=1)
    got = m.panels_by_run(per, genes, "5", 3, m.TO_SOURCE)
    g = genes[(genes.n_t == "5") & (genes.k == 3) & (genes.gamma == 0) & (genes.prefilter_space == "source")]
    want = [(s, b, sorted(x.gene)) for (s, b), x in g.groupby(["split", "subset"])]
    assert got == want and len(got) == 4


def test_split_bootstrap_is_reproducible_paired_and_keeps_subsets_together():
    from rq2_fake_zip import cohort_data, genes_for
    data = cohort_data()
    genes = {c: genes_for(data[c][0], seed=i) for i, c in enumerate(COHORTS)}
    a = m.h4a_cells(data, genes, {c: 40 for c in COHORTS}, n_boot=50)
    b = m.h4a_cells(data, genes, {c: 40 for c in COHORTS}, n_boot=50)
    pd.testing.assert_frame_equal(a, b)
    r = a[(a.cohort == "gse30219") & (a.n_t == "5")].iloc[0]
    assert r.boot_low_mvp == pytest.approx(1.0) and r.boot_low_target_only <= r.phi_target_only + 0.2 <= r.boot_high_target_only + 0.4
    assert r.boot_low <= r.delta <= r.boot_high or r.boot_high - r.boot_low < 1e-9
    assert "boot_low_select_on_source" not in a.columns and r.boot_seed == m.m2.cell_seed("w2split", "gse30219", "5", 2, 3, 5)
    runs = {"x": {2: [(0, 0, ["A", "B"]), (0, 1, ["A", "B"]), (1, 0, ["C", "D"]), (1, 1, ["C", "D"])]}}
    v = m.split_bootstrap(runs, d=10, n_boot=200, seed=1)["x"]
    # mỗi vòng lấy nguyên split: hoặc cả hai split giống nhau (Φ = 1) hoặc mỗi split một lần (Φ của bộ gốc)
    base = m.phi_compact(np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], float), 10)
    assert set(np.round(v, 12)) <= {1.0, round(base, 12)}


def test_main_writes_every_table_and_config(tmp_path, monkeypatch):
    """Y90: main() trên zip giả nhỏ, kiểm tên tệp đầu ra và config.json."""
    import json
    from rq2_fake_zip import cohort_data, patch_run, write_rq2_zip, write_rq4_zip
    main = cohort_data()
    rq2 = write_rq2_zip(tmp_path / "rq2_mvp_fake_results.zip", main)
    rq4 = write_rq4_zip(tmp_path / "rq4_h4b_noihc_fake_results.zip", main)
    patch_run(monkeypatch, m, tmp_path / "out")
    assert m.main(["--rq2", str(rq2), "--rq4", str(rq4), "--n-boot", "10"]) == 0
    out = tmp_path / "out" / "RQ4_tables"
    want = {"rq4_h4b_cells", "rq4_h4b_pooled", "rq4_h4a_cells", "rq4_h4a_pooled", "rq4_decisions", "rq2_ablation_cells", "rq2_ablation_pooled",
            "rq2_noihc_cells", "rq2_noihc_pooled", "rq2_gamma_distribution"}
    assert {p.stem for p in out.glob("*.csv")} == want
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["commit_hash"] == "test000" and cfg["rq2_input_sha256"] and cfg["results_opened_at_utc"].endswith("UTC")
    assert "to_source" in cfg["h4a_arms"] and "w2split" in cfg["h4a_split_bootstrap"]
    h4a = pd.read_csv(out / "rq4_h4a_cells.csv", encoding="utf-8-sig")
    assert {"phi_to_source", "boot_low_to_source", "boot_low", "boot_high"} <= set(h4a.columns)
