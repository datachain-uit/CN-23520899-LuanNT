"""Baseline R của RQ2 (QĐ68, Review 08/11 Mục 8.3): đầu vào cho R, quy tắc panel ptLasso / Trans-GLM, đánh giá như MVP.

R không có trên máy cá nhân: phần R (rq2_r_baselines.R) chỉ kiểm tĩnh (phiên bản ghim, tên tệp khớp với hai script Python); đầu ra R được
giả lập để kiểm toàn bộ phần Python. Test (b) của Review (giai đoạn 2 bằng glmnet trùng ptLasso) đã có ở scripts/r/check_r_questions.R (câu 1c).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
COHORT = "gse37745"
P = 400
PREFILTER = 120


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("kltn_r")
    mp.setenv("KLTN_DATA_ROOT", str(root))
    mp.setenv("KLTN_ALLOW_UNPUSHED", "1")
    mp.setenv("KLTN_ALLOW_UNTRACKED", "1")
    mp.setenv("KLTN_STRICT_N", "0")
    for m in list(sys.modules):
        if m.startswith("scripts.") or m == "scripts" or m.startswith("survey_v2.runlog") or m.startswith("run_") or m.startswith("export_") or m == "provenance":
            del sys.modules[m]
    sys.path.insert(0, str(REPO / "scripts"))
    from survey_v2 import config as C, synthetic
    from survey_v2.data import load_dataset
    from survey_v2.fewshot import build_fewshot_manifest

    rng = np.random.default_rng(2026)
    informative = rng.choice(P, size=15, replace=False)
    genes = np.array([f"G{j:05d}" for j in range(P)])
    genes[informative[:4]] = ["NKX2-1", "TP63", "NAPSA", "KRT5"]
    synthetic.make_dataset(root, C.SOURCE_DATASET, n=160, p=P, effect=0.9, seed=10, informative=informative, genes=genes,
                           with_folds=False, prevalence=0.5)
    synthetic.make_geo_target(root, COHORT, genes, genes[informative], n=90, effect=0.9, seed=22)
    ds = load_dataset(root, COHORT)
    man = build_fewshot_manifest(COHORT, ds.sample_ids, ds.groups, ds.y)
    man.to_csv(ds.data_dir / "fewshot_splits.csv", index=False)
    yield root
    mp.undo()


def _mod(name):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def exported(data_root):
    ex = _mod("export_rq2_inputs")
    assert ex.main(["--datasets", COHORT, "--prefilter", str(PREFILTER), "--n-t", "5", "10"]) == 0
    from survey_v2.runlog import results_dir
    return results_dir("RQ2_r_inputs")


# ------------------------------------------------------------------------------------------------- quy tắc (không cần dữ liệu)
def test_choose_alpha_uses_margin_descending_and_skips_errors():
    ev = _mod("run_rq2_r_eval")
    rows = pd.DataFrame({"alpha": [0.75, 0.5, 0.25], "cvm_min": [1.04, 0.82, 0.86], "error": [np.nan] * 3})
    a, ll = ev.choose_alpha(rows)
    assert a == 0.5 and ll == pytest.approx({0.75: 0.52, 0.5: 0.41, 0.25: 0.43})          # ví dụ của tài liệu ptLasso 09/11
    assert ev.choose_alpha(rows.assign(cvm_min=[1.0, 0.99, 0.985]))[0] == 0.75               # giảm < 0,01 không đổi α
    assert ev.choose_alpha(rows.assign(cvm_min=[1.0, 0.97, 0.90]))[0] == 0.25
    assert ev.choose_alpha(rows.assign(error=["x", np.nan, np.nan]))[0] == 0.5               # α lỗi bị bỏ qua
    assert ev.choose_alpha(rows.assign(error=["x"] * 3))[0] is None


def test_top_k_orders_by_abs_value_and_breaks_ties_by_gene_order():
    ev = _mod("run_rq2_r_eval")
    v = np.array([0.0, -0.7, 0.7, 0.3, 0.0, -0.8])
    assert ev.top_k(v, 3) == [5, 1, 2] and ev.top_k(v, 10) == [5, 1, 2, 3]


def test_panel_fill_uses_ftest_on_target_train_and_flags_count():
    ev = _mod("run_rq2_r_eval")
    from survey_v2.methods import ftest_order
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 6)
    X = rng.normal(size=(12, 6)); X[:, 4] += 3 * y
    panel, n_filled = ev.panel_with_fill(np.array([0, 0.5, 0, 0, 0, 0]), 3, X, y)
    order = [int(j) for j in ftest_order(X, y) if int(j) != 1]
    assert panel == [1, *order[:2]] and n_filled == 2 and panel[1] == 4


def test_same_as_stage1_topk_and_nan_when_s1_too_small():
    ev = _mod("run_rq2_r_eval")
    b1 = np.array([1.2, 0.8, -1.0, 0, 0])
    assert ev.same_as_stage1_topk(b1, [0, 2], 2) == 1.0 and ev.same_as_stage1_topk(b1, [2, 4], 2) == 0.0
    assert np.isnan(ev.same_as_stage1_topk(b1, [0, 1, 2, 3], 4))


# ------------------------------------------------------------------------------------------------- đầu vào cho R
def test_export_matches_mvp_prefilter_folds_and_seeds(data_root, exported):
    from survey_v2 import config as C
    from survey_v2.data import align_to_genes, common_genes, load_dataset, variance_prefilter
    from survey_v2.mvp import inner_splits
    source, ds = load_dataset(data_root, C.SOURCE_DATASET, gene_space=None), load_dataset(data_root, COHORT)
    genes = common_genes(ds, source)
    pre = variance_prefilter(align_to_genes(source, genes), PREFILTER)
    space = (exported / COHORT / "space.txt").read_text(encoding="utf-8").strip()
    assert pd.read_csv(exported / space / "genes.csv").gene.tolist() == list(np.asarray(genes)[pre])
    Z = pd.read_csv(exported / space / "source_z.csv.gz")
    assert Z.shape == (source.n, PREFILTER) and np.allclose(Z.mean(), 0, atol=1e-6) and np.allclose(Z.std(ddof=0), 1, atol=1e-5)
    fold = pd.read_csv(exported / space / "source_foldid.csv").foldid
    assert sorted(fold.unique()) == list(range(1, 11))
    Xt = pd.read_csv(exported / COHORT / "target_x.csv.gz", index_col=0)
    assert np.allclose(Xt.to_numpy(), align_to_genes(ds, genes)[:, pre], rtol=1e-6, atol=1e-6)
    runs = pd.read_csv(exported / COHORT / "runs.csv", dtype={"n_t": str})
    mem = pd.read_csv(exported / COHORT / "membership.csv", dtype={"sample_id": str})
    assert set(runs.n_t) == {"5", "10"} and len(runs) == 40
    cid = C.cohort_id(COHORT)
    r = runs.iloc[0]
    tr = mem[(mem.run_id == r.run_id) & (mem.role == "train")]
    y_of = dict(zip(ds.sample_ids, ds.y))
    y_tr = np.array([y_of[s] for s in tr.sample_id])
    splits = inner_splits(y_tr, C.seed_for(cid, int(r.seed), int(r.subset), int(r.n_t), 9))
    want = np.zeros(len(y_tr), int)
    for f, (_, va) in enumerate(splits[:int(r.n_folds)], start=1):
        want[va] = f
    assert (tr.foldid.to_numpy() == want).all()
    assert r.r_seed == C.seed_for(cid, int(r.seed), int(r.subset), int(r.n_t), 13)


# ------------------------------------------------------------------------------------------------- đánh giá (đầu ra R giả lập)
def _fake_r_output(exported, r_out: Path, rng):
    space = (exported / COHORT / "space.txt").read_text(encoding="utf-8").strip()
    genes = pd.read_csv(exported / space / "genes.csv").gene.tolist()
    beta1 = np.zeros(len(genes)); beta1[[3, 10, 20, 40]] = [1.2, -0.8, 0.5, 0.5]
    r_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"gene": ["(Intercept)", *genes], "beta1": [0.1, *beta1]}).to_csv(r_out / f"stage1_{space}.csv", index=False)
    pd.DataFrame([{"space": space, "n_source": 160, "p": len(genes), "lambda_1se": 0.05, "lambda_min": 0.02, "cvm_1se": 0.6, "size_S1": 4,
                   "seconds": 1.0}]).to_csv(r_out / "stage1_summary.csv", index=False)
    runs = pd.read_csv(exported / COHORT / "runs.csv", dtype={"run_id": str})
    pt, b2, tg, tb = [], [], [], []
    for i, rid in enumerate(runs.run_id):
        for a, cvm in zip((0.75, 0.5, 0.25), (1.04, 0.82, 0.86)):
            pt.append({"run_id": rid, "alpha": a, "lambda_min": 0.1, "cvm_min": cvm, "beta2_0": 0.0, "nnz_beta2": 1, "error": np.nan, "seconds": 0.1})
            b2.append({"run_id": rid, "alpha": a, "gene": genes[50], "beta2": 0.7})
        tg.append({"run_id": rid, "transfer_source_id": "1", "nnz": 1 if i else 0, "intercept": 0.0, "error": np.nan if i else "lỗi giả", "seconds": 2.0})
        if i:
            tb.append({"run_id": rid, "gene": genes[7], "beta": -0.4})
    (r_out / COHORT).mkdir(exist_ok=True)
    for name, rows in (("ptlasso_runs", pt), ("ptlasso_beta2", b2), ("transglm_runs", tg), ("transglm_beta", tb)):
        pd.DataFrame(rows).to_csv(r_out / COHORT / f"{name}.csv", index=False)
    return genes, beta1


def test_evaluation_builds_panels_by_rule_and_evaluates_like_mvp(data_root, exported, tmp_path):
    ev = _mod("run_rq2_r_eval")
    from survey_v2 import config as C
    from survey_v2.classify import evaluate_panel
    from survey_v2.data import load_dataset
    genes, beta1 = _fake_r_output(exported, tmp_path / "r_out", np.random.default_rng(1))
    ds = load_dataset(data_root, COHORT)
    data = ev.load_cohort(exported, tmp_path / "r_out", COHORT)
    rows, gene_rows, diag_rows, score_rows = ev.evaluate_cohort(ds, data, [2, 5], save_scores=True)
    per = pd.DataFrame(rows)
    mvp_cols = ["cohort", "split", "seed", "subset", "n_t", "k", "method", "gamma", "gamma_rule", "prefilter_space", "auroc", "bacc", "mcc", "logloss",
                "brier", "C", "threshold", "n_selected", "exact_k", "n_filled", "n_ihc_in_panel", "select_time_s"]
    assert list(per.columns) == mvp_cols and set(per.method) == {"ptlasso", "transglm"}
    assert len(per[per.method == "ptlasso"]) == 40 * 2 and len(per[per.method == "transglm"]) == 39 * 2      # một lần chạy Trans-GLM lỗi
    g = pd.DataFrame(gene_rows)
    r0 = data["runs"].iloc[0]
    pt2 = g[(g.method == "ptlasso") & (g.k == 2) & (g.split == r0.split) & (g.subset == r0.subset) & (g.n_t == r0.n_t)].sort_values("rank")
    assert pt2.gene.tolist() == [genes[50], genes[3]]                 # β_eff = 0,5·β1 + β2: 0,7 (gen 50) rồi 0,6 (gen 3)
    d = pd.DataFrame(diag_rows)
    assert (d[d.method == "ptlasso"].alpha == 0.5).all() and (d[(d.method == "ptlasso") & (d.k == 2)].same_as_stage1_topk == 0.0).all()
    assert d[(d.method == "transglm") & d.error.notna()].shape[0] == 2
    tg5 = per[(per.method == "transglm") & (per.k == 5)].iloc[0]
    assert tg5.n_filled == 4 and not tg5.exact_k                      # Trans-GLM giả chỉ có 1 gen khác 0
    # đánh giá trùng gọi trực tiếp evaluate_panel với seed của MVP
    row = per[(per.method == "ptlasso") & (per.k == 2)].iloc[0]
    run = data["runs"][(data["runs"].split == row.split) & (data["runs"].subset == row.subset) & (data["runs"].n_t == row.n_t)].iloc[0]
    mem = data["members"][data["members"].run_id == run.run_id]
    idx = {s: i for i, s in enumerate(ds.sample_ids)}
    tr = [idx[s] for s in mem[mem.role == "train"].sample_id]; te = [idx[s] for s in mem[mem.role == "test"].sample_id]
    gi = {gg: j for j, gg in enumerate(ds.genes)}
    cols = [gi[genes[3]], gi[genes[50]]]
    res = evaluate_panel(ds.X[tr][:, cols], ds.y[tr], ds.X[te][:, cols], ds.y[te], classifier="lr_l2",
                         seed=C.seed_for(C.cohort_id(COHORT), int(run.seed), int(run.subset), int(run.n_t), 2))
    assert row.auroc == pytest.approx(res.auroc, abs=1e-12)
    sc = pd.DataFrame(score_rows)
    assert len(sc[(sc.method == "ptlasso") & (sc.k == 2) & (sc.split == row.split) & (sc.subset == row.subset) & (sc.n_t == row.n_t)]) == len(te)


def test_r_script_pins_versions_and_file_names_match_python():
    r = (REPO / "scripts" / "r" / "rq2_r_baselines.R").read_text(encoding="utf-8")
    assert 'ptLasso = "1.0.2"' in r and 'glmtrans = "2.1.0"' in r and "quit(save = \"no\", status = 2L)" in r
    for name in ("source_z.csv.gz", "source_y.csv", "source_foldid.csv", "target_x.csv.gz", "target_y.csv", "runs.csv", "membership.csv", "space.txt"):
        assert name in r and name in (REPO / "scripts" / "export_rq2_inputs.py").read_text(encoding="utf-8")
    ev = (REPO / "scripts" / "run_rq2_r_eval.py").read_text(encoding="utf-8")
    for name in ("stage1_summary.csv", "ptlasso_runs.csv", "ptlasso_beta2.csv", "transglm_runs.csv", "transglm_beta.csv"):
        assert name in r and name in ev
    assert "standardize = FALSE" in r and 'type.measure = "deviance"' in r and "lambda.1se" in r and "lambda.min" in r
    assert "offset = (1 - a) * eta1" in r and "ifelse(in_s1, 1, 1 / a)" in r and 'transfer.source.id = "auto"' in r


def test_kaggle_r_notebook_runs_stage1_first_and_calls_the_three_scripts():
    import json
    nb = json.loads((REPO / "notebooks" / "kaggle_r" / "kaggle_rq2_r_run.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "STAGE = 'stage1'" in code and "REPO_COMMIT = None" in code and "REPO_BRANCH = 'rq2-mvp'" in code
    assert "'r-base==4.5.3'" in code and "'r-glmnet==5.0'" in code and "'KLTN_STRICT_N': '1'" in code
    for s in ("scripts/export_rq2_inputs.py", "scripts/r/rq2_r_baselines.R", "scripts/run_rq2_r_eval.py"):
        assert s in code
    assert "if STAGE == 'all' and r_status == 0" in code and "ignore_patterns('*.csv.gz')" in code
    assert code.index("make_archive") < code.index("raise RuntimeError('export_rq2_inputs.py lỗi")       # lưu zip trước khi báo lỗi


def test_eval_stops_with_code_2_when_library_versions_differ_from_rq2_mvp(capsys):
    ev = _mod("run_rq2_r_eval")
    assert ev.REFERENCE_VERSIONS == {"numpy": "2.1.3", "pandas": "2.3.3", "scipy": "1.16.3", "scikit-learn": "1.6.1"}
    assert ev.version_mismatches(current=dict(ev.REFERENCE_VERSIONS)) == {}
    assert ev.version_mismatches(current={**ev.REFERENCE_VERSIONS, "scikit-learn": "1.7.0"}) == {"scikit-learn": ("1.7.0", "1.6.1")}
    ev.version_mismatches = lambda *a, **k: {"numpy": ("2.2.0", "2.1.3")}
    assert ev.main(["--inputs", "x", "--r-out", "y"]) == 2
    assert "PIP_PINS = {'numpy': '2.1.3'}" in capsys.readouterr().out


def test_kaggle_r_notebook_has_pip_pins_installed_before_imports():
    import json
    nb = json.loads((REPO / "notebooks" / "kaggle_r" / "kaggle_rq2_r_run.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "PIP_PINS = {}" in code and code.index("if PIP_PINS:") < code.index("for mod in ('numpy'")
    assert "eval_status == 2" in code
