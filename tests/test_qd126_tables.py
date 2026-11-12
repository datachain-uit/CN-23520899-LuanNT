"""scripts/make_qd126_tables.py: QĐ126 (Review 11/11) — độ trùng panel, lùi về target-only, phân rã độ ổn định; dữ liệu giả (QĐ126d, Y90).

Viết và commit trước khi mở kết quả QĐ126 (thứ tự mù QĐ69b)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
from rq2_fake_zip import COHORTS, cohort_data, genes_for, patch_run, shuffled_data, write_rq2_zip, write_rq4_zip  # noqa: E402
from test_rq2_tables import synthetic  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("make_qd126_tables_under_test", REPO / "scripts" / "make_qd126_tables.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load()
SPACE = {c: 40 for c in COHORTS}


def _mvp_equals_to_source(data: dict) -> dict:
    """Dòng m5_pos trỏ vào dòng fixed γ = 0 / source của cùng lần chạy: MVP trùng TO_nguồn ở mọi lần chạy (điểm, AUROC, panel)."""
    out = {}
    for c, (per, sc) in data.items():
        src = per[(per.gamma_rule == "fixed") & (per.gamma == 0) & (per.prefilter_space == "source")].copy()
        src["gamma_rule"] = "m5_pos"
        out[c] = (pd.concat([per[per.gamma_rule != "m5_pos"], src], ignore_index=True), sc)
    return out


# ------------------------------------------------------------------------------------------------- (d) test bắt buộc
def test_pi_term_is_exactly_zero_when_mvp_equals_to_source():
    data = _mvp_equals_to_source(cohort_data())
    dec = m.m97.decomposition(data, n_boot=20)["cells"].set_index(["cohort", "n_t", "comparison"]).delta
    for (c, n_t, comp), v in dec.items():
        if comp == "decomp_pi":
            assert v == 0.0
            assert dec[(c, n_t, "decomp_space")] == pytest.approx(dec[(c, n_t, "decomp_total")], abs=1e-15)
    genes = {c: genes_for(data[c][0], seed=i) for i, c in enumerate(COHORTS)}
    st = m.stability_cells(data, genes, SPACE, n_boot=0).set_index(["cohort", "n_t", "comparison"])
    for (c, n_t, comp), r in st.iterrows():
        if comp == "dphi_pi":
            assert r.delta == 0.0
            assert st.loc[(c, n_t, "dphi_space"), "delta"] == pytest.approx(st.loc[(c, n_t, "dphi_total"), "delta"], abs=1e-15)


def test_overlap_is_one_when_panels_coincide():
    assert m.overlap(["A", "B", "C"], ["C", "B", "A"], 3) == 1.0 and m.overlap(["A", "B"], ["C", "A"], 2) == 0.5
    data = cohort_data()
    genes = {c: genes_for(data[c][0], seed=i) for i, c in enumerate(COHORTS)}         # MVP (γ > 0) và select-on-source cùng panel G0..G(k-1)
    runs = m.overlap_runs(data, genes)
    sos = runs[runs.other == "select_on_source"]
    assert len(sos) and (sos.overlap == 1.0).all()
    assert set(runs.n_t) == {"5", "10", "20"} and set(runs.k) == {2, 3, 5}
    tabs = m.overlap_tables(runs)
    assert (tabs["summary"][tabs["summary"].other == "select_on_source"]["mean"] == 1.0).all()
    lab = tabs["labels"].set_index(["other", "n_t"]).label
    assert lab[("select_on_source", "all")] == m.LABEL_A["high"]


def test_fallback_pairs_shuffled_m5_cv_with_target_only_of_the_main_run():
    main = cohort_data()
    shuf = {}
    for i, (c, (per_main, _)) in enumerate(main.items()):
        per, sc = synthetic(cohort=c, n_ts=tuple(sorted(per_main.n_t.unique(), key=int)), ks=(2, 3, 4, 5), seed=90 + i, arms=())
        fixed = per[(per.gamma_rule == "fixed") & (per.gamma == 0.5)]
        cv = fixed.copy(); cv["gamma_rule"] = "m5_cv"                        # m5_cv chọn γ = 0,5; TO của lần xáo KHÁC TO lần chính
        shuf[c] = (pd.concat([per[per.gamma_rule != "m5_cv"], cv], ignore_index=True), sc)
    cells = m.fallback_tables(main, shuf, n_boot=20)["cells"]
    r = cells[(cells.cohort == "gse30219") & (cells.n_t == "5") & (cells.comparison == "m5_cv_shuffled_vs_target_only")].iloc[0]
    sel = lambda f, q: f.query(q).set_index(["split", "subset", "k"]).auroc  # noqa: E731
    a = sel(shuf["gse30219"][0], "gamma_rule == 'm5_cv' and n_t == '5' and k in [2, 3, 5]")
    b = sel(main["gse30219"][0], "gamma_rule == 'fixed' and gamma == 0 and prefilter_space == 'target' and n_t == '5' and k in [2, 3, 5]")
    assert r.delta == pytest.approx((a - b).groupby(level=["split", "subset"]).mean().mean(), abs=1e-12)
    b_wrong = sel(shuf["gse30219"][0], "gamma_rule == 'fixed' and gamma == 0 and prefilter_space == 'target' and n_t == '5' and k in [2, 3, 5]")
    assert abs((a - b_wrong).groupby(level=["split", "subset"]).mean().mean() - r.delta) > 1e-6


def test_m5_cv_choosing_gamma_zero_gives_exact_fallback_and_label():
    main = cohort_data()
    shuf = shuffled_data(main)                                               # m5_cv chọn γ = 0 ở mọi lần chạy, TO chép từ lần chính
    fb = m.fallback_tables(main, shuf, n_boot=20)
    cv = fb["cells"][fb["cells"].comparison == "m5_cv_shuffled_vs_target_only"]
    assert (cv.delta == 0).all() and (cv.se == 0).all()
    lab = m.fallback_labels(fb["pooled"])
    assert lab.set_index("n_t").label["all"] == m.LABEL_B[True]
    g0 = m.gamma0_share(main, shuf)
    assert (g0[g0.source_labels == "shuffled"].share_gamma0 == 1.0).all()
    assert (g0[g0.source_labels == "true"].share_gamma0 == 1.0).all()        # synthetic: m5_cv của lần chính cũng là γ = 0 / target


def test_check_shuffled_refuses_missing_or_duplicated_m5_cv_rows():
    main = cohort_data()
    shuf = shuffled_data(main)
    assert m.check_shuffled(main, shuf)["gse30219"] == int(((main["gse30219"][0].gamma_rule == "m5_pos")).sum())
    per, sc = shuf["gse30219"]
    drop = per[per.gamma_rule == "m5_cv"].index[0]
    with pytest.raises(SystemExit, match="tập ô"):
        m.check_shuffled(main, {**shuf, "gse30219": (per.drop(index=drop), sc)})
    dup = pd.concat([per, per[per.gamma_rule == "m5_cv"].iloc[:1]], ignore_index=True)
    with pytest.raises(SystemExit, match="trùng"):
        m.check_shuffled(main, {**shuf, "gse30219": (dup, sc)})


# ------------------------------------------------------------------------------------------------- quy tắc đọc
def test_overlap_label_thresholds_and_scenarios():
    assert m._label_a([0.8, 0.9, 1.0]) == m.LABEL_A["high"]
    assert m._label_a([0.5, 0.1, 0.3]) == m.LABEL_A["low"]
    assert m._label_a([0.79, 0.9, 1.0]) == m.LABEL_A["mid"]
    assert m.scenario(m.LABEL_A["high"], m.LABEL_B[False]) == "A"
    assert m.scenario(m.LABEL_A["mid"], m.LABEL_B[True]) == "B"
    assert m.scenario(m.LABEL_A["low"], m.LABEL_B[False]) == "C"
    assert m.scenario(m.LABEL_A["mid"], m.LABEL_B[False]) == "không khớp kịch bản nào"


def test_fallback_rule_needs_both_pools_above_minus_margin():
    pooled = pd.DataFrame([{"comparison": "m5_cv_shuffled_vs_target_only", "n_t": n, "pool_set": "gpl570", "cohorts": "x",
                            "lower_ivw": li, "lower_eq": le} for n, li, le in (("5", -0.005, -0.009), ("10", -0.002, -0.011), ("20", 0.0, 0.0))])
    pooled["noninferior"] = (pooled.lower_ivw > -m.MARGIN) & (pooled.lower_eq > -m.MARGIN)
    pooled["label"] = pooled.noninferior.map(m.LABEL_B)
    lab = m.fallback_labels(pooled).set_index("n_t").label
    assert lab["5"] == m.LABEL_B[True] and lab["10"] == m.LABEL_B[False] and lab["all"] == m.LABEL_B[False]


def test_fallback_with_zero_se_cohort_uses_equal_weights_and_flags_it():
    main = cohort_data()
    shuf = shuffled_data(main)
    p = m.fallback_tables(main, shuf, n_boot=20)["pooled"]
    cv = p[(p.comparison == "m5_cv_shuffled_vs_target_only") & (p.pool_set == "gpl570")]
    assert cv.ivw_undefined.all() and cv.noninferior.all() and cv.label.str.contains("IVW không xác định").all()
    pos = p[p.comparison == "m5_pos_shuffled_vs_target_only"]
    assert (pos.label == "").all()


def test_stability_components_add_up_and_match_h4a_delta():
    data = cohort_data()
    genes = {c: genes_for(data[c][0], seed=i) for i, c in enumerate(COHORTS)}
    st = m.stability_cells(data, genes, SPACE, n_boot=30).set_index(["cohort", "n_t", "comparison"])
    h4a = m.m4.h4a_cells(data, genes, SPACE, n_boot=30).set_index(["cohort", "n_t"])
    for (c, n_t), r in h4a.iterrows():
        tot = st.loc[(c, n_t, "dphi_total")]
        assert tot.delta == pytest.approx(st.loc[(c, n_t, "dphi_space"), "delta"] + st.loc[(c, n_t, "dphi_pi"), "delta"], abs=1e-12)
        assert tot.delta == pytest.approx(r.delta, abs=1e-12) and tot.se == pytest.approx(r.se, abs=1e-12)
        assert tot.boot_low == pytest.approx(r.boot_low, abs=1e-12) and tot.boot_high == pytest.approx(r.boot_high, abs=1e-12)
        assert tot.phi_select_on_source == pytest.approx(r.phi_select_on_source, nan_ok=True)
    pooled = m.stability_pooled(st.reset_index())
    assert set(pooled.pool_set) == {"gpl570", "all4"} and (pooled.family == m.DESCRIPTIVE).all()


# ------------------------------------------------------------------------------------------------- main() (Y90)
def test_main_writes_every_table_and_config(tmp_path, monkeypatch):
    main = cohort_data()
    rq2 = write_rq2_zip(tmp_path / "rq2_mvp_fake_results.zip", main)
    rq4 = write_rq4_zip(tmp_path / "rq4_h4b_noihc_fake_results.zip", main)
    patch_run(monkeypatch, m, tmp_path / "out")
    assert m.main(["--rq2", str(rq2), "--rq4", str(rq4), "--n-boot", "10"]) == 0
    out = tmp_path / "out" / "RQ2_qd126_tables"
    assert {p.stem for p in out.glob("*.csv")} == {
        "qd126a_overlap_runs", "qd126a_overlap_summary", "qd126a_overlap_by_k", "qd126a_overlap_by_gamma", "qd126a_labels",
        "qd126b_fallback_cells", "qd126b_fallback_pooled", "qd126b_gamma0_share", "qd126c_stability_cells", "qd126c_stability_pooled",
        "qd126_reading"}
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["commit_hash"] == "test000" and cfg["seed_key"] == "qd126" and len(cfg["input_sha256"]) == 2
    assert cfg["results_opened_at_utc"].endswith("UTC") and set(cfg["shuffled_m5_cv_rows"]) == set(COHORTS)
    read = pd.read_csv(out / "qd126_reading.csv", encoding="utf-8-sig")
    assert read[read.analysis == "bang5"].label.item() in {"kịch bản A", "kịch bản B", "kịch bản C", "kịch bản không khớp kịch bản nào"}
