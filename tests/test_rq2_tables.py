"""scripts/make_rq2_tables.py (Review 08/11 Mục 8.4, QĐ69): bootstrap hai tầng, gộp IVW / trọng số bằng nhau, Holm, quy tắc H2, H2b, H2c.

Viết và commit trước khi mở kết quả GEO của RQ2 (QĐ69b): mọi test chạy trên dữ liệu giả, trừ một test trên zip pilot TCGA (dữ liệu phát triển).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load():
    spec = importlib.util.spec_from_file_location("make_rq2_tables_under_test", REPO / "scripts" / "make_rq2_tables.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load()


# --------------------------------------------------------------------------------------------- dữ liệu giả
def synthetic(cohort="gse30219", n_splits=2, n_subsets=2, n_ts=("5", "10"), ks=(2, 3, 4, 5), n_pos=6, n_neg=5, seed=0,
              arms=("d3", "ihc"), mvp_shift=0.0, chosen=None):
    """per_run + scores đúng định dạng của run_rq2_mvp.py. AUROC trong per_run tính bằng roc_auc_score trên đúng điểm đã ghi.

    `chosen[(split, subset, n_t, k)]` là γ mà m5_pos chọn (mặc định 0,5); điểm của MVP lệch thêm `mvp_shift` theo nhãn."""
    rng = np.random.default_rng(seed)
    per, scores = [], []
    chosen = chosen or {}
    for split in range(n_splits):
        test_ids = [f"{cohort}_s{split}_p{i}" for i in range(n_pos)] + [f"{cohort}_s{split}_n{i}" for i in range(n_neg)]
        y = np.array([1] * n_pos + [0] * n_neg)
        for subset in range(n_subsets):
            for n_t in n_ts:
                base = {"cohort": cohort, "split": split, "seed": 42 + split, "subset": subset, "n_t": n_t}

                def add(method, k, gamma, rule, space, shift=0.0, write_scores=True):
                    s = np.round(rng.normal(size=len(y)) + y * (0.8 + shift), 1)        # làm tròn để có điểm hoà
                    row = {**base, "k": k, "method": method, "gamma": gamma, "gamma_rule": rule, "prefilter_space": space,
                           "auroc": float(roc_auc_score(y, s)), "bacc": 0.5, "mcc": 0.0, "logloss": 0.7, "brier": 0.25}
                    per.append(row)
                    if write_scores:
                        scores.extend({**base, "k": k, "method": method, "gamma": gamma, "prefilter_space": space,
                                       "sample_id": sid, "y_true": int(t), "score": float(v)} for sid, t, v in zip(test_ids, y, s))
                    return row

                for k in ks:
                    fixed = {}
                    for gamma, space in ((0.0, "target"), (0.0, "source"), (0.5, "source"), (1.0, "source"), (2.0, "source")):
                        fixed[(gamma, space)] = add("mvp", k, gamma, "fixed", space, shift=mvp_shift if gamma > 0 else 0.0)
                    g = chosen.get((split, subset, n_t, k), 0.5)
                    pos = dict(fixed[(g, "source")]); pos["gamma_rule"] = "m5_pos"; per.append(pos)
                    cv = dict(fixed[(0.0, "target")]); cv["gamma_rule"] = "m5_cv"; per.append(cv)
                    if "d3" in arms and n_t in ("5", "10", "20"):
                        for method in ("select_on_source", "pooled"):
                            add(method, k, np.nan, "none", "source")
                if "ihc" in arms:
                    for method, k in (("ihc_2", 2), ("ihc_4", 4)):
                        add(method, k, np.nan, "none", "none")
    return pd.DataFrame(per), pd.DataFrame(scores)


# --------------------------------------------------------------------------------------------- AUROC có trọng số
def test_pair_matrix_auroc_matches_sklearn_with_ties():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.5, 0.5, 0.5, 0.2, 0.9, 0.1])
    M = m.pair_matrix(s, y)
    assert M.shape == (3, 4)
    assert M.mean() == pytest.approx(roc_auc_score(y, s), abs=1e-15)


def test_counts_auroc_equals_auroc_on_duplicated_samples():
    rng = np.random.default_rng(1)
    y = np.array([1] * 5 + [0] * 4)
    s = np.round(rng.normal(size=9) + y, 1)
    cpos, cneg = np.array([2, 0, 1, 1, 1]), np.array([0, 3, 1, 0])
    M = m.pair_matrix(s, y)
    got = cpos @ M @ cneg / (cpos.sum() * cneg.sum())
    idx = np.r_[np.repeat(np.arange(5), cpos), 5 + np.repeat(np.arange(4), cneg)]
    assert got == pytest.approx(roc_auc_score(y[idx], s[idx]), abs=1e-15)


# --------------------------------------------------------------------------------------------- ghép điểm từng nhánh
def test_mvp_scores_follow_gamma_chosen_by_m5_pos_and_target_only_uses_target_space():
    chosen = {(0, 0, "5", 2): 2.0, (1, 1, "5", 2): 1.0}
    per, sc = synthetic(chosen=chosen, arms=())
    runs = m.collect_runs(per, sc, "gse30219", "5", "mvp", "target_only", (2,))
    by_run = {(r.split, r.subset): r for r in runs}
    want = per[(per.gamma_rule == "fixed") & (per.gamma == 2.0) & (per.split == 0) & (per.subset == 0) & (per.n_t == "5") & (per.k == 2)].auroc.item()
    assert by_run[(0, 0)].auroc_a[2] == pytest.approx(want)
    to = per[(per.gamma_rule == "fixed") & (per.gamma == 0) & (per.prefilter_space == "target") & (per.split == 0) & (per.subset == 0) & (per.n_t == "5") & (per.k == 2)]
    assert by_run[(0, 0)].auroc_b[2] == pytest.approx(to.auroc.item())
    src = per[(per.gamma_rule == "fixed") & (per.gamma == 0) & (per.prefilter_space == "source") & (per.split == 0) & (per.subset == 0) & (per.n_t == "5") & (per.k == 2)]
    assert to.auroc.item() != src.auroc.item()          # dữ liệu giả phân biệt được hai không gian gen của γ = 0


def test_point_estimate_equals_mean_of_per_run_differences():
    per, sc = synthetic(ks=(2, 3, 5))
    cell = m.bootstrap_cell(m.collect_runs(per, sc, "gse30219", "10", "mvp", "select_on_source", (2, 3, 5)), n_boot=50, seed=1)
    a = per[per.gamma_rule == "m5_pos"].query("n_t == '10'").set_index(["split", "subset", "k"]).auroc
    b = per[per.method == "select_on_source"].query("n_t == '10'").set_index(["split", "subset", "k"]).auroc
    assert cell["delta"] == pytest.approx((a - b).groupby(level=["split", "subset"]).mean().mean(), abs=1e-12)
    assert cell["n_runs"] == 4 and cell["n_splits"] == 2


def test_scores_disagreeing_with_per_run_auroc_are_refused():
    per, sc = synthetic(arms=())
    sc.loc[sc.index[0], "score"] += 5.0
    with pytest.raises(ValueError, match="AUROC"):
        m.collect_runs(per, sc, "gse30219", "5", "mvp", "target_only", (2, 3, 4, 5))


def test_missing_arm_rows_are_refused():
    per, sc = synthetic(arms=())
    with pytest.raises(ValueError, match="pooled"):
        m.collect_runs(per, sc, "gse30219", "5", "mvp", "pooled", (2,))


def test_test_set_must_be_the_same_for_every_subset_of_a_split():
    """Tầng 2 dùng chung test_T trong một split: nếu hai tập con của cùng split có test_T khác nhau thì giả định sai, phải dừng."""
    per, sc = synthetic(arms=())
    mask = (sc.split == 0) & (sc.subset == 1)
    sc.loc[mask, "sample_id"] = sc.loc[mask, "sample_id"] + "_x"
    with pytest.raises(ValueError, match="test_T"):
        m.collect_runs(per, sc, "gse30219", "5", "mvp", "target_only", (2,))


# --------------------------------------------------------------------------------------------- bootstrap hai tầng
def test_tier2_resample_is_shared_within_a_split_and_stratified():
    """QĐ69a (Y71): hai lần chạy cùng split có cùng ma trận D cho cùng giá trị ở MỌI vòng; khác split thì khác; mỗi vòng giữ số mẫu từng lớp."""
    rng = np.random.default_rng(3)
    D = rng.normal(size=(6, 5))
    runs = [m.Run(0, 0, D, {}, {}), m.Run(0, 1, D, {}, {}), m.Run(1, 0, D, {}, {})]
    cell, vals, counts = m.bootstrap_cell(runs, n_boot=200, seed=7, return_draws=True)
    assert np.array_equal(vals[:, 0], vals[:, 1])
    assert not np.allclose(vals[:, 0], vals[:, 2])
    for split, (cpos, cneg) in counts.items():
        assert (cpos.sum(axis=1) == 6).all() and (cneg.sum(axis=1) == 5).all()
    assert cell["se"] > 0 and cell["n_boot"] == 200


def test_bootstrap_is_reproducible_and_seed_comes_from_crc32_key():
    per, sc = synthetic()
    runs = m.collect_runs(per, sc, "gse30219", "5", "mvp", "target_only", (2, 3, 5))
    s1 = m.cell_seed("gse30219", "5", "mvp-target_only")
    assert s1 == m.cell_seed("gse30219", "5", "mvp-target_only") and s1 != m.cell_seed("gse30219", "10", "mvp-target_only")
    a, b = m.bootstrap_cell(runs, n_boot=100, seed=s1), m.bootstrap_cell(runs, n_boot=100, seed=s1)
    assert a == b


def test_bootstrap_se_reflects_both_tiers():
    """SE phải > 0 cả khi mọi lần chạy giống hệt nhau (chỉ còn tầng 2), và khi test_T chỉ gồm điểm hoà (chỉ còn tầng 1)."""
    rng = np.random.default_rng(5)
    D = rng.normal(size=(6, 5))
    same = [m.Run(s, u, D, {}, {}) for s in range(3) for u in range(2)]
    assert m.bootstrap_cell(same, n_boot=300, seed=1)["se"] > 0
    const = [m.Run(s, u, np.full((6, 5), 0.01 * (s * 2 + u)), {}, {}) for s in range(3) for u in range(2)]
    cell = m.bootstrap_cell(const, n_boot=300, seed=1)
    assert cell["se"] > 0 and cell["delta"] == pytest.approx(np.mean([0.01 * i for i in range(6)]))


# --------------------------------------------------------------------------------------------- gộp và quy tắc
def test_pool_ivw_and_equal_weights_hand_numbers():
    cells = pd.DataFrame({"cohort": ["a", "b", "c"], "delta": [0.02, 0.01, -0.01], "se": [0.01, 0.02, 0.02]})
    p = m.pool(cells, margin=0.0)
    w = np.array([1 / 0.01**2, 1 / 0.02**2, 1 / 0.02**2])
    d_ivw = float((w * cells.delta).sum() / w.sum()); se_ivw = float(1 / np.sqrt(w.sum()))
    assert p["delta_ivw"] == pytest.approx(d_ivw) and p["se_ivw"] == pytest.approx(se_ivw)
    assert p["delta_eq"] == pytest.approx(0.02 / 3) and p["se_eq"] == pytest.approx(np.sqrt(0.01**2 + 2 * 0.02**2) / 3)
    assert p["lower_ivw"] == pytest.approx(d_ivw - 1.6448536269514722 * se_ivw)
    q = float((w * (cells.delta - d_ivw) ** 2).sum())
    assert p["i2"] == pytest.approx(max(0.0, (q - 2) / q))
    from scipy.stats import norm
    assert p["p_ivw"] == pytest.approx(norm.sf(d_ivw / se_ivw))
    assert m.pool(cells, margin=0.01)["p_ivw"] == pytest.approx(norm.sf((d_ivw + 0.01) / se_ivw))
    assert p["n_cohorts"] == 3 and p["w_a"] == pytest.approx(w[0] / w.sum())


def _pooled(rows):
    return pd.DataFrame(rows)


def test_h2_rule():
    def table(p, d_ivw, d_eq):
        return _pooled([{"n_t": n, "p_ivw": pv, "delta_ivw": a, "delta_eq": e} for n, pv, a, e in zip(("5", "10", "20"), p, d_ivw, d_eq)])
    out, holm = m.decide_h2(table([0.004, 0.2, 0.3], [0.02, 0.01, 0.0], [0.01, 0.01, 0.0]))
    assert out == "ung_ho" and holm == pytest.approx([0.012, 0.4, 0.4])
    assert m.decide_h2(table([0.004, 0.2, 0.3], [0.02, 0.01, 0.0], [-0.001, 0.01, 0.0]))[0] == "khong_ket_luan"   # bản bằng nhau ngược dấu
    assert m.decide_h2(table([0.6, 0.7, 0.9], [0.0, -0.01, -0.02], [0.01, 0.0, 0.0]))[0] == "bac_bo"
    assert m.decide_h2(table([0.6, 0.7, 0.9], [0.001, -0.01, -0.02], [0.0, 0.0, 0.0]))[0] == "khong_ket_luan"


def test_h2c_rule():
    def table(p, lower_eq):
        keys = [(s, n) for s in ("select_on_source", "pooled") for n in ("5", "10", "20")]
        return _pooled([{"comparison": s, "n_t": n, "p_ivw": pv, "lower_eq": le} for (s, n), pv, le in zip(keys, p, lower_eq)])
    ok = [-0.005] * 6
    out, holm = m.decide_h2c(table([0.001] * 6, ok))
    assert out == "ung_ho" and holm == pytest.approx([0.006] * 6)
    assert m.decide_h2c(table([0.001] * 5 + [0.2], ok))[0] == "ung_ho_mot_phan"
    assert m.decide_h2c(table([0.001] * 6, ok[:5] + [-0.02]))[0] == "ung_ho_mot_phan"   # mọi kiểm định bác bỏ H0 nhưng một cận dưới bản bằng nhau ≤ −0,01
    assert m.decide_h2c(table([0.3] * 6, ok))[0] == "khong_ung_ho"


# --------------------------------------------------------------------------------------------- đầu-cuối trên dữ liệu giả
def _inputs(cohorts=("gse10245", "gse30219", "gse37745", "gse42127")):
    data = {}
    for i, c in enumerate(cohorts):
        n_ts = ("5", "10") if c == "gse10245" else ("5", "10", "20", "40")
        data[c] = synthetic(cohort=c, n_ts=n_ts, ks=(2, 3, 4, 5), seed=10 + i, mvp_shift=0.6)
    return data


def test_build_produces_every_table_and_pools_only_cohorts_that_have_the_level():
    tables = m.build(_inputs(), n_boot=40)
    cells, pooled, h2b, decisions = tables["cells"], tables["pooled"], tables["h2b"], tables["decisions"]
    h2 = cells[cells.comparison == "target_only"]
    assert set(h2.n_t) == {"5", "10", "20", "40"} and set(h2[h2.n_t == "20"].cohort) == {"gse30219", "gse37745", "gse42127"}
    main20 = pooled[(pooled.comparison == "target_only") & (pooled.n_t == "20") & (pooled.pool_set == "gpl570")].iloc[0]
    assert main20.n_cohorts == 2                                   # GSE10245 chỉ có n_T ∈ {5, 10}
    assert set(pooled.pool_set) == {"gpl570", "all4"}
    assert set(cells[cells.comparison.isin(["select_on_source", "pooled"])].n_t) == {"5", "10", "20"}
    assert set(h2b.comparison) == {"ihc_2", "ihc_4"} and set(h2b.n_t) == {"5", "10"} and "lower_ivw" in h2b and "noninferior_both" in h2b
    assert set(decisions.hypothesis) == {"H2", "H2c"}
    assert decisions.set_index("hypothesis").loc["H2", "outcome"] == "ung_ho"      # MVP được đẩy lên rõ trong dữ liệu giả
    assert pooled.loc[pooled.comparison == "target_only", "p_holm"].notna().sum() == 3


def test_gate_must_pass_before_any_statistic(tmp_path):
    """QĐ70: cổng bằng nhau thiếu hoặc có dòng không ok thì không tính thống kê."""
    report = tmp_path / "rq2_equality_report.csv"
    with pytest.raises(SystemExit, match="cổng"):
        m.check_gate(report)
    pd.DataFrame({"status": ["ok", "mismatch"]}).to_csv(report, index=False)
    with pytest.raises(SystemExit, match="cổng"):
        m.check_gate(report)
    pd.DataFrame({"status": ["ok", "ok"]}).to_csv(report, index=False)
    assert m.check_gate(report) == 2


def test_load_from_zip_and_directory_give_the_same_frames(tmp_path):
    data = _inputs(("gse30219",))
    root = tmp_path / "RQ2_mvp" / "gse30219"
    root.mkdir(parents=True)
    per, sc = data["gse30219"]
    per.to_csv(root / "rq2_mvp_per_run.csv", index=False)
    sc.to_csv(root / "rq2_mvp_scores.csv", index=False)
    pd.DataFrame({"status": ["ok"]}).to_csv(tmp_path / "RQ2_mvp" / "rq2_equality_report.csv", index=False)
    z = tmp_path / "rq2_mvp_x_results.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in (tmp_path / "RQ2_mvp").rglob("*.csv"):
            zf.write(f, "results/survey_v2/" + f.relative_to(tmp_path).as_posix())
    from_dir, rep_dir = m.load_inputs(tmp_path / "RQ2_mvp", ["gse30219"])
    from_zip, rep_zip = m.load_inputs(z, ["gse30219"])
    pd.testing.assert_frame_equal(from_dir["gse30219"][0], from_zip["gse30219"][0])
    pd.testing.assert_frame_equal(from_dir["gse30219"][1], from_zip["gse30219"][1])
    assert list(rep_dir.status) == list(rep_zip.status) == ["ok"]
    assert from_dir["gse30219"][0].n_t.map(type).eq(str).all()          # "pool" và số đều đọc thành chuỗi


# --------------------------------------------------------------------------------------------- pilot (dữ liệu phát triển)
PILOT = os.environ.get("KLTN_RQ2_PILOT_ZIP")


@pytest.mark.skipif(not PILOT, reason="đặt KLTN_RQ2_PILOT_ZIP = zip của phiên rq2_early 9a15860 (không có trong repo)")
def test_pilot_zip_point_estimates_match_per_run():
    """Pilot TCGA (9a15860, trước QĐ51): chưa có cột method/prefilter_space trong scores và chưa có m5_pos; so m5_cv với target-only."""
    with zipfile.ZipFile(PILOT) as zf:
        read = lambda suffix: pd.read_csv(zf.open(next(n for n in zf.namelist() if n.endswith("tcga_early/" + suffix))), dtype={"n_t": str})
        per, sc = read("rq2_mvp_per_run.csv"), read("rq2_mvp_scores.csv")
    for n_t in ("5", "10"):
        runs = m.collect_runs(per, sc, "tcga_early", n_t, "mvp", "target_only", (2, 4, 5), mvp_rule="m5_cv")
        cell = m.bootstrap_cell(runs, n_boot=200, seed=m.cell_seed("tcga_early", n_t, "pilot"))
        a = per[(per.gamma_rule == "m5_cv") & (per.n_t == n_t) & per.k.isin([2, 4, 5])].set_index(["split", "subset", "k"]).auroc
        b = per[(per.gamma_rule == "fixed") & (per.gamma == 0) & (per.n_t == n_t) & per.k.isin([2, 4, 5])].set_index(["split", "subset", "k"]).auroc
        assert cell["delta"] == pytest.approx((a - b).groupby(level=["split", "subset"]).mean().mean(), abs=1e-12)
        assert cell["n_runs"] == 20 and np.isfinite(cell["se"]) and cell["se"] > 0


# --------------------------------------------------------------------------------------------- baseline R (ptLasso, Trans-GLM; mô tả)
def r_synthetic(cohort="gse30219", n_splits=2, n_subsets=2, n_ts=("5", "10"), ks=(2, 3, 4, 5), n_pos=6, n_neg=5, seed=50):
    """rq2_r_per_run.csv, rq2_r_scores.csv, rq2_r_diagnostics.csv như run_rq2_r_eval.py, cùng test_T với synthetic(cohort, n_pos, n_neg)."""
    rng = np.random.default_rng(seed)
    per, sc, diag = [], [], []
    for split in range(n_splits):
        test_ids = [f"{cohort}_s{split}_p{i}" for i in range(n_pos)] + [f"{cohort}_s{split}_n{i}" for i in range(n_neg)]
        y = np.array([1] * n_pos + [0] * n_neg)
        for subset in range(n_subsets):
            for n_t in n_ts:
                base = {"cohort": cohort, "split": split, "seed": 42 + split, "subset": subset, "n_t": n_t}
                for method in ("ptlasso", "transglm"):
                    for k in ks:
                        s = np.round(rng.normal(size=len(y)) + 0.6 * y, 1)
                        per.append({**base, "k": k, "method": method, "gamma": np.nan, "gamma_rule": "none", "prefilter_space": "source",
                                    "auroc": float(roc_auc_score(y, s)), "bacc": 0.5, "mcc": 0.0, "logloss": 0.7, "brier": 0.25})
                        sc.extend({**base, "k": k, "method": method, "gamma": np.nan, "prefilter_space": "source", "sample_id": sid,
                                   "y_true": int(t), "score": float(v)} for sid, t, v in zip(test_ids, y, s))
                        d = {**base, "method": method, "k": k}
                        if method == "ptlasso":
                            d.update(alpha=0.5, same_as_stage1_topk=float((split + subset + k) % 3 == 0), size_S1=28)
                        diag.append(d)
    return pd.DataFrame(per), pd.DataFrame(sc), pd.DataFrame(diag)


def test_r_baselines_are_descriptive_and_leave_h2_h2b_h2c_unchanged():
    data = _inputs()
    base = m.build(data, n_boot=40)
    r = {c: r_synthetic(cohort=c, n_ts=("5", "10") if c == "gse10245" else ("5", "10", "20", "40"), seed=60 + i) for i, c in enumerate(data)}
    full = m.build(data, n_boot=40, r_data=r)
    for name in ("cells", "pooled", "h2b", "decisions"):
        old = base[name]
        new = full[name]
        if name in ("cells", "pooled"):
            new = new[~new.comparison.isin(m.R_ARMS)].reset_index(drop=True)
            old = old[[c for c in old.columns]].reset_index(drop=True)
            new = new[old.columns]
        pd.testing.assert_frame_equal(old.reset_index(drop=True), new.reset_index(drop=True))
    b = full["baselines"]
    assert set(b.comparison) == set(m.R_ARMS) and set(b.n_t) == {"5", "10", "20", "40"} and b.p_holm.isna().all()
    assert (b.family == m.R_FAMILY).all() and {"ci95_ivw_low", "ci95_ivw_high", "ci95_eq_low", "ci95_eq_high"} <= set(b.columns)
    row = b.iloc[0]
    assert row.ci95_ivw_low == pytest.approx(row.delta_ivw - 1.959963984540054 * row.se_ivw)


def test_r_comparison_pairs_on_the_same_test_set_as_mvp():
    per, sc = synthetic(arms=())
    rp, rs, _ = r_synthetic()
    runs = m.collect_runs(pd.concat([per, rp]), pd.concat([sc, rs]), "gse30219", "5", "mvp", "ptlasso", (2, 3, 5))
    a = rp[(rp.method == "ptlasso") & (rp.n_t == "5") & rp.k.isin([2, 3, 5])].set_index(["split", "subset", "k"]).auroc
    assert runs[0].auroc_b == {k: pytest.approx(a.loc[(runs[0].split, runs[0].subset, k)]) for k in (2, 3, 5)}
    bad = rs.copy()
    bad.loc[bad.split == 0, "sample_id"] = bad.loc[bad.split == 0, "sample_id"] + "_x"
    with pytest.raises(ValueError, match="test_T"):
        m.collect_runs(pd.concat([per, rp]), pd.concat([sc, bad]), "gse30219", "5", "mvp", "ptlasso", (2,))


def test_degeneracy_table_by_cohort_n_t_k_and_tuning_budget():
    _, _, diag = r_synthetic()
    t = m.degeneracy_table({"gse30219": diag})
    assert set(t.columns) >= {"cohort", "n_t", "k", "n_runs", "same_as_stage1_topk_rate"} and len(t) == 2 * 4
    row = t[(t.n_t == "5") & (t.k == 3)].iloc[0]
    sub = diag[(diag.method == "ptlasso") & (diag.n_t == "5") & (diag.k == 3)]
    assert row.n_runs == 4 and row.same_as_stage1_topk_rate == pytest.approx(sub.same_as_stage1_topk.mean())
    budget = m.tuning_budget()
    assert set(budget.method) == {"mvp_m5_pos", "target_only", "select_on_source", "pooled", "ihc", "ptlasso", "transglm"}
    assert {"tuned_on_L_T", "selection_stage", "classifier_stage"} <= set(budget.columns)


def test_load_r_inputs_from_zip_matches_directory(tmp_path):
    rp, rs, rd = r_synthetic()
    root = tmp_path / "RQ2_r" / "gse30219"
    root.mkdir(parents=True)
    rp.to_csv(root / "rq2_r_per_run.csv", index=False); rs.to_csv(root / "rq2_r_scores.csv", index=False); rd.to_csv(root / "rq2_r_diagnostics.csv", index=False)
    z = tmp_path / "rq2_r_all_x_results.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in root.glob("*.csv"):
            zf.write(f, "rq2_r/results_survey_v2/RQ2_r/gse30219/" + f.name)
    a, b = m.load_r_inputs(tmp_path / "RQ2_r", ["gse30219"]), m.load_r_inputs(z, ["gse30219"])
    for x, y in zip(a["gse30219"], b["gse30219"]):
        pd.testing.assert_frame_equal(x, y)
    assert a["gse30219"][0].n_t.map(type).eq(str).all()


def test_cli_accepts_r_input_option():
    args = m.parse_args(["--input", "a.zip", "--r-input", "b.zip"])
    assert str(args.r_input) == "b.zip" and m.parse_args([]).r_input is None
