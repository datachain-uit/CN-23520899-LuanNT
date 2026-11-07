"""Hai script tổng hợp không chạy lại mô hình (Review 07/11): Bảng 3 của pilot (bước 7) và độ nhạy RQ1 (QĐ53)."""

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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- summarize_rq2_early --------------------------------------------------------------------------------------------------
def _pilot_frames():
    """Một ô (n_T = 5, k = 2), hai lần chạy; số viết tay.

    Lần 0: log-loss {0: 0,50; 0,5: 0,60; 1: 0,595; 2: 0,592} → m5_cv = 0 (mọi γ > 0 đều tệ hơn); m5_pos = 0,5 (1 và 2 giảm chưa tới 0,01).
    Lần 1: log-loss {0: 0,60; 0,5: 0,50; 1: 0,495; 2: 0,492} → m5_cv = 0,5; m5_pos = 0,5.
    """
    auroc = {0: {0.0: 0.80, 0.5: 0.90, 1.0: 0.91, 2.0: 0.92}, 1: {0.0: 0.70, 0.5: 0.95, 1.0: 0.96, 2.0: 0.97}}
    logloss = {0: {0.0: 0.50, 0.5: 0.60, 1.0: 0.595, 2.0: 0.592}, 1: {0.0: 0.60, 0.5: 0.50, 1.0: 0.495, 2.0: 0.492}}
    base = lambda s: {"split": 0, "subset": s, "n_t": "5", "k": 2}
    per, path, sos = [], [], []
    for s in (0, 1):
        for g, a in auroc[s].items():
            per.append({**base(s), "method": "mvp", "gamma": g, "gamma_rule": "fixed", "auroc": a})
            path.append({**base(s), "gamma": g, "oof_logloss": logloss[s][g], "selected": False})
    per.append({**base(0), "method": "mvp", "gamma": 0.0, "gamma_rule": "m5_cv", "auroc": 0.80})
    per.append({**base(1), "method": "mvp", "gamma": 0.5, "gamma_rule": "m5_cv", "auroc": 0.95})
    sos = [{**base(0), "method": "select_on_source", "auroc": 0.85}, {**base(1), "method": "select_on_source", "auroc": 0.96}]
    return pd.DataFrame(per), pd.DataFrame(path), pd.DataFrame(sos)


def test_rules_summary_matches_hand_numbers():
    m = _load("summarize_rq2_early")
    summary, pairs, gammas = m.summarize(*_pilot_frames())
    cell = summary[summary.n_t == "5"].iloc[0]
    assert cell.target_only == pytest.approx((0.80 + 0.70) / 2)
    assert cell.m5_cv == pytest.approx((0.80 + 0.95) / 2)
    assert cell.m5_pos == pytest.approx((0.90 + 0.95) / 2)
    assert cell.gamma_1 == pytest.approx((0.91 + 0.96) / 2)
    assert cell.select_on_source == pytest.approx((0.85 + 0.96) / 2)
    row = pairs.set_index("comparison").loc["m5_cv - select_on_source"]
    assert row.mean_diff == pytest.approx(((0.80 - 0.85) + (0.95 - 0.96)) / 2) and (row.wins, row.losses, row.n_pairs) == (0, 2, 2)
    row = pairs.set_index("comparison").loc["m5_cv - target_only"]
    assert (row.wins, row.losses, row.ties) == (1, 0, 1)
    row = pairs.set_index("comparison").loc["m5_cv chose gamma=0: m5_cv - gamma_0.5"]
    assert row.n_pairs == 1 and row.mean_diff == pytest.approx(0.80 - 0.90)
    counts = gammas.set_index(["rule", "gamma"]).n_runs.to_dict()
    assert counts == {("m5_cv", 0.0): 1, ("m5_cv", 0.5): 1, ("m5_pos", 0.5): 2}


def test_float_noise_counts_as_tie():
    """Pilot thật có một cặp chênh 1,1e-16 giữa hai AUROC thực chất bằng nhau: phải đếm là hoà, nếu không số thắng là 122 thay vì 121 của Review 07/11."""
    m = _load("summarize_rq2_early")
    row = m._pair("x", pd.Series([0.9 + 1.1e-16, 0.91, 0.8]), pd.Series([0.9, 0.90, 0.9]))
    assert (row["wins"], row["losses"], row["ties"], row["n_pairs"]) == (1, 1, 1, 3)


def test_mean_row_averages_cells_not_runs():
    """Hàng cuối của Bảng 3 là trung bình 8 ô. Hai ô khác số lần chạy phải có trọng số bằng nhau."""
    m = _load("summarize_rq2_early")
    per, path, sos = _pilot_frames()
    extra = per[per.subset == 0].assign(n_t="10")                  # ô thứ hai chỉ một lần chạy
    summary, _, _ = m.summarize(pd.concat([per, extra]), pd.concat([path, path[path.subset == 0].assign(n_t="10")]),
                                pd.concat([sos, sos[sos.subset == 0].assign(n_t="10")]))
    assert summary[summary.n_t == "all"].iloc[0].target_only == pytest.approx((0.75 + 0.80) / 2)


def test_recomputed_m5_cv_disagreeing_with_file_is_refused():
    """Bảng dựa trên việc tính lại quy tắc từ đường γ, nên lệch với dòng m5_cv trong tệp thì phải dừng."""
    m = _load("summarize_rq2_early")
    per, path, sos = _pilot_frames()
    per.loc[(per.gamma_rule == "m5_cv") & (per.subset == 0), "gamma"] = 0.5
    with pytest.raises(ValueError, match="m5_cv"):
        m.summarize(per, path, sos)


def test_load_pilot_reads_members_from_zip(tmp_path):
    m = _load("summarize_rq2_early")
    per, path, sos = _pilot_frames()
    z = tmp_path / "rq2_early_results.zip"
    prefix = "results/survey_v2/RQ2_mvp_early/tcga_early/"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(prefix + "rq2_mvp_per_run.csv", per.to_csv(index=False))
        zf.writestr(prefix + "rq2_mvp_gamma_path.csv", path.to_csv(index=False))
        zf.writestr(prefix + "rq2_early_select_on_source.csv", sos.to_csv(index=False))
    loaded = m.load_pilot(z)
    assert len(loaded[0]) == len(per) and loaded[0].n_t.iloc[0] == "5" and len(loaded[1]) == len(path) and len(loaded[2]) == len(sos)


# --- make_rq1_sensitivity -------------------------------------------------------------------------------------------------
def _per_fold(cohorts, diff, seed=0):
    """Bảng theo fold 5×5: B0 và l1_lr với chênh lệch cho trước theo k; thêm dòng RF để chắc chắn lọc model = lr_l2."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in cohorts:
        for r in range(5):
            for f in range(5):
                b0 = 0.9 + rng.normal(0, 0.005)
                rows.append({"cohort": c, "repeat": r, "fold": f, "method": "b0", "k": -1, "model": "lr_l2", "auroc": b0})
                rows.append({"cohort": c, "repeat": r, "fold": f, "method": "b0", "k": -1, "model": "rf", "auroc": 0.0})
                for k, d in diff.items():
                    rows.append({"cohort": c, "repeat": r, "fold": f, "method": "l1_lr", "k": k, "model": "lr_l2", "auroc": b0 + d + rng.normal(0, 0.002)})
                    rows.append({"cohort": c, "repeat": r, "fold": f, "method": "l1_lr", "k": k, "model": "rf", "auroc": 0.0})
    return pd.DataFrame(rows)


DIFF = {50: 0.0, 25: 0.0, 10: 0.0, 5: -0.05, 4: -0.05, 3: -0.05, 2: -0.05, 1: -0.05}
COHORTS3 = ["gse10245", "gse30219", "gse37745"]


def _rq1_root(tmp_path):
    root = tmp_path / "results"
    for folder, cohorts, diff in (("E1_classification", [*COHORTS3, "gse42127"], DIFF),
                                  ("E1_classification_noIHC", COHORTS3, {**DIFF, 10: -0.05, 25: 0.0})):
        frame = _per_fold(cohorts, diff)
        for c in cohorts:
            (root / folder / c).mkdir(parents=True)
            frame[frame.cohort == c].to_csv(root / folder / c / "classification_per_fold.csv", index=False)
    return root


def test_sensitivity_rows_have_expected_k_star(tmp_path):
    m = _load("make_rq1_sensitivity")
    summary, by_k, main_table = m.build(_rq1_root(tmp_path))
    rows = summary.set_index("config")
    assert list(rows.index) == ["main", "four_cohorts", "no_ihc"]
    assert (rows.loc["main", "k_star_both"], rows.loc["main", "k_star_ivw"]) == (10, 10)
    assert rows.loc["four_cohorts", "cohorts"] == "gse10245,gse30219,gse37745,gse42127" and rows.loc["four_cohorts", "k_star_both"] == 10
    assert rows.loc["no_ihc", "k_star_both"] == 25                  # dữ liệu giả của nhánh này tụt ở k = 10
    assert rows.loc["main", "lb_ivw_k10"] > -0.02 and rows.loc["no_ihc", "lb_ivw_k10"] < -0.02
    assert rows.loc["main", "k_sequence"] == "50>25>10>5>4>3>2>1"
    assert sorted(json.loads(rows.loc["main", "weights_k10"])) == COHORTS3
    assert sum(json.loads(rows.loc["main", "weights_k10"]).values()) == pytest.approx(1.0)
    assert set(by_k.config) == {"main", "four_cohorts", "no_ihc"} and len(by_k[by_k.config == "main"]) == 8
    assert len(main_table) == 8


def test_rf_rows_do_not_enter_k_star(tmp_path):
    """Dòng model = rf có AUROC 0 sẽ kéo k* về None nếu lọc sai."""
    m = _load("make_rq1_sensitivity")
    summary, _, _ = m.build(_rq1_root(tmp_path))
    assert summary.set_index("config").loc["main", "k_star_both"] is not None and summary.lb_eq_k10.notna().all()


def test_reference_check_counts_matching_rows(tmp_path):
    m = _load("make_rq1_sensitivity")
    _, _, main_table = m.build(_rq1_root(tmp_path))
    assert m.check_reference(main_table, main_table.copy()) == (8, 8, [])
    changed = main_table.copy()
    changed.loc[changed.k == 10, "lb_ivw"] += 1e-6
    n_ok, n_all, bad = m.check_reference(main_table, changed)
    assert (n_ok, n_all) == (7, 8) and bad == [("l1_lr", 10, "lb_ivw")]
    assert m.check_reference(main_table, main_table[main_table.k != 10])[:2] == (7, 8)     # thiếu dòng cũng là không khớp


@pytest.mark.skipif(not os.environ.get("KLTN_RQ1_OFFICIAL_ROOT"), reason="cần results/survey_v2 của panels, ablation_phaseB và tables (không có trong repo)")
def test_main_config_reproduces_official_table_and_review_numbers():
    """QĐ53: cấu hình chính tái lập 92/92 dòng của e1_kstar_pooled_by_k.csv; số của Review 07/11 Bảng 5."""
    m = _load("make_rq1_sensitivity")
    root = Path(os.environ["KLTN_RQ1_OFFICIAL_ROOT"])
    summary, _, main_table = m.build(root)
    reference = pd.read_csv(root / "tables" / "e1_kstar_pooled_by_k.csv")
    assert m.check_reference(main_table, reference) == (92, 92, [])
    rows = summary.set_index("config")
    assert (rows.loc["main", "k_star_both"], rows.loc["main", "k_star_ivw"]) == (10, 10)
    assert (rows.loc["four_cohorts", "k_star_both"], rows.loc["four_cohorts", "k_star_ivw"]) == (10, 3)
    assert (rows.loc["no_ihc", "k_star_both"], rows.loc["no_ihc", "k_star_ivw"]) == (10, 2)
    assert round(rows.loc["four_cohorts", "lb_ivw_k10"], 4) == -0.0084 and round(rows.loc["four_cohorts", "lb_eq_k10"], 4) == -0.0082
    assert round(rows.loc["no_ihc", "lb_ivw_k10"], 4) == -0.0079 and round(rows.loc["no_ihc", "lb_eq_k10"], 4) == -0.0120
