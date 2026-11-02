"""scripts/make_survey_tables.py — v3.1.7 (Review 02/11/2026, QĐ30–QĐ31): seed bootstrap tái lập, cờ E7 theo Holm, bảng ablation IHC."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("make_survey_tables_under_test", REPO / "scripts" / "make_survey_tables.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mk = _load()


def test_stable_seed_is_deterministic_across_processes():
    """QĐ30: seed không phụ thuộc PYTHONHASHSEED (hash() của Python bị ngẫu nhiên hoá theo tiến trình)."""
    code = ("import sys; sys.path.insert(0, %r); import importlib.util; "
            "spec = importlib.util.spec_from_file_location('m', %r); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            "print(m._stable_seed('gse37745', 'l1_lr', 10, 'pool'), m._boot_ci([0.9, 0.8, 0.95, 0.7, 0.85], seed=m._stable_seed('a', 1)))"
            % (str(REPO), str(REPO / "scripts" / "make_survey_tables.py")))
    outs = set()
    for hs in ("0", "1", "random"):
        env = {**os.environ, "PYTHONHASHSEED": hs}
        outs.add(subprocess.check_output([sys.executable, "-c", code], env=env, text=True, timeout=120).strip())
    assert len(outs) == 1, outs
    assert mk._stable_seed("a", 1) != mk._stable_seed("a", 2) and 0 <= mk._stable_seed("x") < 2**31


def test_e7_flags_follow_holm_not_bootstrap():
    """QĐ30: positive/negative_transfer = p Wilcoxon-Holm < 0,05 và dấu của diff_mean; CI bootstrap chỉ mô tả."""
    rng = np.random.default_rng(0)
    rows = []
    for n_t in (5, 10, 20):
        for run in range(20):
            split, subset = run // 4, run % 4
            base = 0.8 + 0.01 * rng.standard_normal()
            rows.append({"cohort": "c", "split": split, "subset": subset, "n_t": str(n_t), "k": 5, "strategy": "target_only", "auroc": base})
            shift = {5: 0.05, 10: 0.0005, 20: -0.05}[n_t]                 # rõ dương / mờ / rõ âm
            rows.append({"cohort": "c", "split": split, "subset": subset, "n_t": str(n_t), "k": 5, "strategy": "pooled",
                         "auroc": base + shift + 0.002 * rng.standard_normal()})
    tab = mk.e7_table(pd.DataFrame(rows)).set_index("n_t")
    assert bool(tab.loc["5", "positive_transfer"]) and not bool(tab.loc["5", "negative_transfer"])
    assert bool(tab.loc["20", "negative_transfer"]) and not bool(tab.loc["20", "positive_transfer"])
    assert tab.loc["5", "p_wilcoxon_holm"] < 0.05 and tab.loc["20", "p_wilcoxon_holm"] < 0.05
    assert "ci_excludes_zero" in tab.columns
    # cờ đúng là hàm của p_holm và dấu, không phải của CI
    assert (tab.positive_transfer == ((tab.p_wilcoxon_holm < 0.05) & (tab.diff_mean > 0))).all()
    assert (tab.negative_transfer == ((tab.p_wilcoxon_holm < 0.05) & (tab.diff_mean < 0))).all()


def test_e1_ablation_ihc_pairs_folds_and_counts_ihc_genes():
    """QĐ31: Δ = không IHC − đầy đủ, ghép theo (cohort, repeat, fold, method, k); B0 ghép theo method (k khác 4 gen);
    random-k lấy trung bình các lần rút; số gen IHC trung bình trong panel đầy đủ."""
    rng = np.random.default_rng(1)
    full, noihc, genes = [], [], []
    for rep in range(2):
        for fold in range(3):
            for m, k in (("f_test", 2), ("f_test", 5), ("ihc_2", 2)):
                a = 0.9 + 0.01 * rng.standard_normal()
                full.append({"cohort": "c", "repeat": rep, "fold": fold, "method": m, "k": k, "classifier": "lr_l2", "auroc": a})
                if m != "ihc_2":
                    noihc.append({"cohort": "c", "repeat": rep, "fold": fold, "method": m, "k": k, "classifier": "lr_l2", "auroc": a - 0.02})
                    for r in range(k):
                        genes.append({"cohort": "c", "repeat": rep, "fold": fold, "method": m, "k": k, "gene": "NKX2-1" if r == 0 else f"G{r}"})
            full.append({"cohort": "c", "repeat": rep, "fold": fold, "method": "b0", "k": 400, "classifier": "lr_l2", "auroc": 0.95})
            noihc.append({"cohort": "c", "repeat": rep, "fold": fold, "method": "b0", "k": 396, "classifier": "lr_l2", "auroc": 0.95})
            for draw in range(2):
                full.append({"cohort": "c", "repeat": rep, "fold": fold, "method": "random_k", "k": 2, "classifier": "lr_l2", "auroc": 0.7 + 0.1 * draw})
                noihc.append({"cohort": "c", "repeat": rep, "fold": fold, "method": "random_k", "k": 2, "classifier": "lr_l2", "auroc": 0.7 + 0.1 * draw})
    abl = mk.e1_ablation_ihc(pd.DataFrame(full), pd.DataFrame(noihc), pd.DataFrame(genes), "lr_l2")
    assert set(abl.method) == {"f_test", "b0", "random_k"}            # IHC-2 không có bản không IHC
    f2 = abl[(abl.method == "f_test") & (abl.k == 2)].iloc[0]
    assert abs(f2.delta_mean + 0.02) < 1e-9 and f2.n_folds == 6 and abs(f2.ihc_genes_per_panel - 1.0) < 1e-9
    assert abl[abl.method == "b0"].k.iloc[0] == "all" and abs(abl[abl.method == "b0"].delta_mean.iloc[0]) < 1e-12
    assert abs(abl[abl.method == "random_k"].delta_mean.iloc[0]) < 1e-12 and np.isnan(abl[abl.method == "random_k"].ihc_genes_per_panel.iloc[0])
    assert {"delta_ci_low", "delta_ci_high", "auroc_full", "auroc_noihc"} <= set(abl.columns)
