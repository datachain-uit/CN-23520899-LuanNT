"""Chạy đầu-cuối mọi script survey v3 trên dữ liệu tổng hợp nhỏ (KLTN_DATA_ROOT tạm), kiểm tra tệp đầu ra và bất biến.

Chuỗi: build_e1_repeated_splits → build_fewshot_splits → E0 → E1 pha A (+ wrapper) → E1 pha B → E1b → E7 → E2/E3 →
E4/E5 → make_survey_tables → sanity Y10. Chạy chậm (~2–4 phút). Bỏ qua bằng: pytest -m "not slow".
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow

COHORT = "gse37745"
SMALL = "gse10245"


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def data_root(tmp_path_factory, monkeypatch_module):
    root = tmp_path_factory.mktemp("kltn")
    monkeypatch_module.setenv("KLTN_DATA_ROOT", str(root))
    monkeypatch_module.setenv("KLTN_ALLOW_UNPUSHED", "1")
    monkeypatch_module.setenv("KLTN_ALLOW_UNTRACKED", "1")
    for m in list(sys.modules):
        if m.startswith("scripts.") or m == "scripts" or m.startswith("survey_v2.runlog") or m.startswith("run_survey") or m == "provenance":
            del sys.modules[m]
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    from survey_v2 import synthetic
    synthetic.make_all(root, n=120, p=400, effect=0.9, n_target=90)
    return root


def _load(name):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def test_01_splits(data_root):
    _load("build_e1_repeated_splits").main([])
    _load("build_fewshot_splits").main([])
    man = pd.read_csv(data_root / "experiment-data-geo-target" / COHORT / "repeated_splits.csv")
    assert man[["repeat", "fold"]].drop_duplicates().shape[0] == 25
    fs = pd.read_csv(data_root / "experiment-data-geo-target" / COHORT / "fewshot_splits.csv", dtype={"n_t": str})
    assert {"5", "10", "pool"} <= set(fs.n_t)
    assert (data_root / "experiment-data-geo-target" / "fewshot_splits_manifest.json").exists()


def test_02_e0_signal_check(data_root):
    _load("run_survey_e0_signal_check").main(["--datasets", COHORT, "--n-perm", "8", "--n-random", "6", "--prefilter", "150", "--n-jobs", "1"])
    t = pd.read_csv(data_root / "results" / "survey_v2" / "E0" / COHORT / "e0_signal_check.csv")
    assert len(t) == 1 and t.p_perm.iloc[0] >= 1 / 9 - 1e-9 and 0 <= t.auroc_real.iloc[0] <= 1
    assert t.signal_present.iloc[0] in (True, False)


def test_03_e1_phase_a_b(data_root):
    a = _load("run_e1_feature_selection")
    a.main(["--datasets", COHORT, SMALL, "--methods", "f_test", "l1_lr", "stability_selection", "ktsp", "ihc_2", "ihc_4",
            "--k", "1", "2", "3", "4", "5", "10", "--repeats", "0", "--prefilter", "150"])
    d = data_root / "results" / "survey_v2" / "E1_feature_selection" / COHORT
    sel = pd.read_csv(d / "selection_per_fold.csv")
    assert set(sel.method) == {"f_test", "l1_lr", "stability_selection", "ktsp", "ihc_2", "ihc_4"}
    assert (sel[sel.method == "ktsp"].set_index("k").n_selected.groupby(level=0).first().reindex([3, 5]) == 0).all()
    genes = pd.read_csv(d / "selected_genes.csv")
    g = genes[(genes.method == "f_test") & (genes.fold == 0)]
    assert set(g[g.k == 2].gene) <= set(g[g.k == 5].gene)
    assert set(genes[genes.method == "ihc_2"].gene) == {"NKX2-1", "TP63"}
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    assert cfg["gene_space"]["gene_space"].startswith("intersection") and cfg["fixed_panels"]["ihc_4"][2] == "NAPSA"
    # wrapper riêng
    w = _load("run_e1_wrappers_all_cohorts")
    w.run(["--datasets", COHORT, "--methods", "ga", "--repeats", "0", "--k", "5", "--pop", "6", "--generations", "2", "--prefilter", "80"])
    gw = pd.read_csv(data_root / "results" / "survey_v2" / "E1_wrapper_feature_selection" / COHORT / "selection_per_fold.csv")
    assert (gw.n_eval > 0).all() and (gw.n_selected == 5).all()
    # pha B: LR-L2 chính, phụ ở k ∈ SECONDARY_K, B0, random-k; không có best-by-test
    b = _load("run_e1_classification")
    b.main(["--datasets", COHORT, "--repeats", "0", "--secondary", "--random-draws", "3", "--prefilter", "150"])
    db = data_root / "results" / "survey_v2" / "E1_classification" / COHORT
    pf = pd.read_csv(db / "classification_per_fold.csv")
    assert {"b0", "random_k", "f_test", "ga", "ihc_2"} <= set(pf.method)
    assert set(pf.model) == {"lr_l2", "rf", "svm_linear"}
    assert set(pf[pf.model != "lr_l2"].k) <= {5, 25}
    assert pf.auroc.between(0, 1).all() and pf[pf.model == "lr_l2"].brier.notna().all()
    assert pf[pf.model == "svm_linear"].logloss.isna().all()
    assert not (db / "best_classifier_by_panel.csv").exists()
    rk = pd.read_csv(db / "e1_random_k.csv")
    assert rk.groupby(["fold", "k"]).size().eq(3).all()
    n_before = len(pf)
    b.main(["--datasets", COHORT, "--repeats", "0", "--secondary", "--random-draws", "3", "--prefilter", "150"])   # resume: không nhân đôi
    assert len(pd.read_csv(db / "classification_per_fold.csv")) == n_before
    # v3.1.6 (QĐ25): --redo-methods chỉ tính lại một phương pháp, phần còn lại giữ nguyên (checkpoint / dòng cũ)
    ck = d / "checkpoints"
    other_before = {f.name: f.read_bytes() for f in ck.glob("*.csv") if "_l1_lr." not in f.name}
    a.main(["--datasets", COHORT, "--methods", "f_test", "l1_lr", "stability_selection", "ktsp", "ihc_2", "ihc_4", "--redo-methods", "l1_lr",
            "--k", "1", "2", "3", "4", "5", "10", "--repeats", "0", "--prefilter", "150"])
    assert {f.name: f.read_bytes() for f in ck.glob("*.csv") if "_l1_lr." not in f.name} == other_before
    sel2 = pd.read_csv(d / "selection_per_fold.csv")
    assert set(sel2.method) == set(sel.method) and len(sel2) == len(sel)
    cfg2 = json.loads((d / "config.json").read_text(encoding="utf-8"))
    assert cfg2["redo_methods"] == ["l1_lr"] and cfg2["checkpoints_reused_from_commit"] == cfg["commit_hash"]
    with pytest.raises(SystemExit):
        a.main(["--datasets", COHORT, "--methods", "f_test", "--redo-methods", "l1_lr", "--repeats", "0", "--prefilter", "150"])
    b.main(["--datasets", COHORT, "--repeats", "0", "--secondary", "--no-references", "--methods", "l1_lr", "--redo-methods", "l1_lr",
            "--prefilter", "150"])
    pf2 = pd.read_csv(db / "classification_per_fold.csv")
    assert len(pf2) == n_before and set(pf2.method) == set(pf.method)
    assert pf2.groupby(["repeat", "fold", "method", "k", "model"]).size().max() == 1
    assert json.loads((db / "config.json").read_text(encoding="utf-8"))["redo_methods"] == ["l1_lr"]


def test_04_e1b_and_e7(data_root):
    _load("run_survey_e1b_learning_curve").main(["--datasets", COHORT, "--methods", "f_test", "l1_lr", "ihc_2", "--k", "2", "5",
                                                 "--n-t", "5", "10", "pool", "--prefilter", "120", "--random-draws", "2", "--n-jobs", "1"])
    d = data_root / "results" / "survey_v2" / "E1b" / COHORT
    pr = pd.read_csv(d / "e1b_per_run.csv", dtype={"n_t": str})
    assert set(pr.method) == {"f_test", "l1_lr", "ihc_2", "random_k"} and set(pr.n_t) == {"5", "10", "pool"}
    assert pr.groupby(["split", "subset", "n_t", "method", "k"]).size().max() == 1
    # M5: n_T <= 10 -> C = 1 cố định
    assert (pr[(pr.n_t == "5") & (pr.method == "f_test")].C == 1.0).all()
    st = pd.read_csv(d / "e1b_stability.csv")
    assert (st.n_runs == 20).all() and st.nogueira.between(-1, 1.0001).all()
    # v3.1.6: --redo-methods l1_lr -> số dòng không đổi, các dòng f_test giữ nguyên
    f_before = pr[pr.method == "f_test"].sort_values(["split", "subset", "n_t", "k"]).auroc.tolist()
    _load("run_survey_e1b_learning_curve").main(["--datasets", COHORT, "--methods", "l1_lr", "--redo-methods", "l1_lr", "--k", "2", "5",
                                                 "--n-t", "5", "10", "pool", "--prefilter", "120", "--random-draws", "2", "--n-jobs", "1"])
    pr2 = pd.read_csv(d / "e1b_per_run.csv", dtype={"n_t": str})
    assert len(pr2) == len(pr) and set(pr2.method) == set(pr.method)
    assert pr2[pr2.method == "f_test"].sort_values(["split", "subset", "n_t", "k"]).auroc.tolist() == f_before
    assert json.loads((d / "config.json").read_text(encoding="utf-8"))["redo_methods"] == ["l1_lr"]
    _load("run_survey_e7_transfer_naive").main(["--datasets", COHORT, "--k", "2", "5", "--n-t", "5", "--prefilter", "120", "--n-jobs", "1"])
    e7 = pd.read_csv(data_root / "results" / "survey_v2" / "E7" / COHORT / "e7_per_run.csv", dtype={"n_t": str})
    assert set(e7.strategy) == {"source_only", "select_on_source", "pooled", "target_only"}
    assert e7.groupby(["split", "subset", "n_t", "strategy", "k"]).size().max() == 1 and e7.auroc.between(0, 1).all()
    so = e7[e7.strategy == "source_only"]
    assert so.groupby(["split", "k"]).auroc.nunique().max() == 1      # zero-shot không phụ thuộc tập con
    cfg = json.loads((data_root / "results" / "survey_v2" / "E7" / COHORT / "config.json").read_text(encoding="utf-8"))
    assert len(cfg["source_panels"]["2"]) == 2 and cfg["gene_space_size"] > 100


def test_05_e2_e3_timing(data_root):
    t = _load("run_survey_e2_e3_timing")
    sys.argv = ["run_survey_e2_e3_timing.py", "--methods", "f_test", "ktsp", "--repeats", "1", "--n-grid", "30", "60", "all",
                "--p-grid", "50", "100", "all", "--k", "4", "10", "--wrappers", "--isolate-memory"]
    import survey_v2.config as C
    C.WRAPPER_POP, C.WRAPPER_GENERATIONS = 4, 2
    C.E3_WRAPPER_BUDGETS = ((4, 2), (5, 2))
    t.main()
    out = data_root / "results" / "survey_v2" / "E2_E3"
    cost = pd.read_csv(out / "e2_cost_point.csv")
    assert set(cost.method) == {"f_test", "ktsp", "ga", "bpso"} and set(cost.k) == {4, 10} and (cost.wall_s > 0).all()
    assert "peak_mb" not in cost.columns and "maxrss_mb" not in cost.columns                      # v3.1.6 (QĐ27)
    assert cost.maxrss_isolated_mb.notna().all() and (cost.maxrss_child_mb < 1500).all()         # bộ nhớ của chính tiến trình con
    exps = pd.read_csv(out / "e3_exponents.csv")
    assert set(exps.method) == {"f_test", "ktsp"}
    cfg = json.loads((out / "config.json").read_text())
    assert (out / "machine_config.json").exists() and cfg["k_e3"] == 4 and cfg["timing_tracemalloc"] is False
    if sys.platform.startswith("linux"):
        assert cfg["peak_source"] == "VmHWM" and (cost.peak_source == "VmHWM").all()
    assert (out / "e3_wrapper_evals.csv").exists()                                                 # có wrapper -> có tệp


def test_06_e4_e5_stability(data_root):
    s = _load("run_survey_e4_e5_stability")
    s.main(["--datasets", COHORT, "--methods", "f_test", "l1_lr", "stability_selection", "--k", "2", "5", "10", "--m", "4",
            "--fractions", "0.5", "0.7", "--prefilter", "120", "--wrapper-k", "10", "--pop", "4", "--generations", "2", "--n-jobs", "1"])
    out = data_root / "results" / "survey_v2" / "E4_E5" / COHORT
    stab = pd.read_csv(out / "e4_stability.csv")
    assert set(stab.method) == {"f_test", "l1_lr", "stability_selection", "random_k"} and (stab.M == 4).all()
    assert stab[stab.method == "f_test"].nogueira.mean() > stab[stab.method == "random_k"].nogueira.mean()
    e5a = pd.read_csv(out / "e5a_redundancy.csv")
    assert set(e5a.rho) == {0.5, 0.8, 0.95}
    e5b = pd.read_csv(out / "e5b_seed_stability.csv")
    assert {"rf_importance", "stability_selection", "ga", "bpso"} <= set(e5b.method)
    assert (out / "e5c_overlap_k10.csv").exists()
    # v3.1.6: --redo-methods l1_lr -> E4/E5a của l1_lr chọn lại, f_test giữ nguyên
    g_before = pd.read_csv(out / "e4_selected_genes.csv")
    s.main(["--datasets", COHORT, "--methods", "l1_lr", "--redo-methods", "l1_lr", "--k", "2", "5", "10", "--m", "4",
            "--fractions", "0.5", "0.7", "--prefilter", "120", "--skip-e5b", "--n-jobs", "1"])
    g_after = pd.read_csv(out / "e4_selected_genes.csv")
    assert len(g_after) == len(g_before) and set(g_after.method) == set(g_before.method)
    key = ["fraction", "subsample", "method", "k", "rank"]
    pd.testing.assert_frame_equal(g_before[g_before.method == "f_test"].sort_values(key).reset_index(drop=True),
                                  g_after[g_after.method == "f_test"].sort_values(key).reset_index(drop=True))
    stab2 = pd.read_csv(out / "e4_stability.csv")
    assert len(stab2) == len(stab) and json.loads((out / "config.json").read_text(encoding="utf-8"))["redo_methods"] == ["l1_lr"]


def test_07_tables_and_sanity(data_root):
    mk = _load("make_survey_tables")
    mk.main(["--e6-fraction", "0.7"])
    out = data_root / "results" / "survey_v2" / "tables"
    for f in ("e0_signal_check.csv", "e1_summary.csv", "e1_tests_vs_random_b0.csv", "e1_kstar_pooled.csv", "e1_kstar_per_cohort.csv",
              "e1b_learning_curves.csv", "e7_transfer_table.csv", "e2_cost_table.csv", "e4_nogueira_table.csv", "e6_tradeoff.csv",
              "provenance_manifest.json", "e1_auroc_table.tex"):
        assert (out / f).exists(), f
    tests = pd.read_csv(out / "e1_tests_vs_random_b0.csv")
    assert {"p_nb_random_holm", "p_nb_b0_holm"} <= set(tests.columns) and tests.p_nb_random.between(0, 1).all()
    ks = pd.read_csv(out / "e1_kstar_pooled.csv")
    assert "k_star_both" in ks.columns and "k_sequence_used" in ks.columns
    e7 = pd.read_csv(out / "e7_transfer_table.csv")
    assert {"diff_ci_low", "p_wilcoxon_holm", "negative_transfer"} <= set(e7.columns)
    figs = list((data_root / "results" / "survey_v2" / "figures").glob("fig_e1b_*.png")) + list((data_root / "results" / "survey_v2" / "figures").glob("fig_e7_*.png"))
    assert len(figs) >= 2
    man = json.loads((out / "provenance_manifest.json").read_text(encoding="utf-8"))
    assert man["inputs"] and all("sha256" in v for v in man["inputs"].values())
    sa = _load("run_sanity_auroc")
    rc = sa.main(["--repeats", "0", "--prefilter", "150"])
    summ = pd.read_csv(data_root / "results" / "survey_v2" / "sanity" / "sanity_summary.csv")
    assert set(summ.method) == {"f_test", "l1_lr", "rf_importance"} and rc in (0, 1)
