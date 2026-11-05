"""MVP bài 2 (Review 03/11 Mục 7.3): khung bước 0–4 và test (a) γ = 0 ≡ E1b target-only.

Dữ liệu tổng hợp nhỏ; `prefilter` < p nên lọc sơ bộ thật sự có tác dụng: nếu γ = 0 lọc trên nguồn thay vì trên L_T, test (a) sẽ hỏng.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

COHORT = "gse37745"
P = 400
PREFILTER = 120
K = [2, 5]


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("kltn_mvp")
    mp.setenv("KLTN_DATA_ROOT", str(root))
    mp.setenv("KLTN_ALLOW_UNPUSHED", "1")
    mp.setenv("KLTN_ALLOW_UNTRACKED", "1")
    mp.setenv("KLTN_STRICT_N", "0")                                 # runner Kaggle đặt = 1; dữ liệu giả không có số mẫu thật
    for m in list(sys.modules):
        if m.startswith("scripts.") or m == "scripts" or m.startswith("survey_v2.runlog") or m.startswith("run_") or m == "provenance":
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


def _load(name):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def _setup(root):
    """Nguồn, đích, ngữ cảnh, π và các lần chạy n_T ∈ {5, 10} cho đúng một cohort giả."""
    from survey_v2 import config as C
    from survey_v2.data import load_dataset
    from survey_v2.fewshot import fewshot_runs
    from survey_v2.mvp import source_stability
    from survey_v2.transfer import TransferContext

    source, ds = load_dataset(root, C.SOURCE_DATASET, gene_space=None), load_dataset(root, COHORT)
    cid = C.cohort_id(COHORT)
    ctx = TransferContext(source, ds, K, selector="l1_lr", prefilter=PREFILTER, seed=C.seed_for(cid, 0, 0, 7))
    pi, c_q = source_stability(ctx, cid)
    man = pd.read_csv(ds.data_dir / "fewshot_splits.csv", dtype={"n_t": str})
    runs = [r for r in fewshot_runs(man) if r[3] in ("5", "10")][:6]
    return ds, ctx, cid, pi, c_q, runs


def test_penalty_weights_formula():
    from survey_v2.mvp import penalty_weights

    pi = np.array([0.0, 0.95, 1.0])
    assert np.allclose(penalty_weights(pi, 1.0), [20.0, 1.0, 1 / 1.05])
    assert np.allclose(penalty_weights(pi, 2.0), [400.0, 1.0, 1 / 1.05**2])
    assert np.array_equal(penalty_weights(pi, 0.0), np.ones(3))


def test_weights_steer_selection_towards_stable_genes():
    """Hỏng nếu trọng số bị triệt tiêu, vd. gọi select() vì hàm đó z-score lại X_w."""
    from survey_v2.mvp import select_weighted

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 60))
    y = np.tile([0, 1], 15)
    pi = np.zeros(60)
    pi[[5, 6]] = 1.0
    assert set(select_weighted(X, y, pi, 2.0, [2], seed=1).panels[2].tolist()) == {5, 6}
    assert set(select_weighted(X, y, pi, 0.0, [2], seed=1).panels[2].tolist()) != {5, 6}


def test_a_gamma0_matches_e1b_target_only(data_root):
    """Test (a): γ = 0 cho đúng panel và chỉ số của target-only L1-LR trong E1b, sai khác < 1e-12."""
    e1b, mvp = _load("run_survey_e1b_learning_curve"), _load("run_rq2_mvp")
    ds, ctx, cid, pi, _, runs = _setup(data_root)
    compared = 0
    for split, seed, subset, n_t, tr, te in runs:
        e_rows, e_genes, _ = e1b.run_one(ds, cid, split, seed, subset, n_t, tr, te, ["l1_lr"], K,
                                         SimpleNamespace(prefilter=PREFILTER, random_draws=0))
        m_rows, m_genes, _ = mvp.run_one(ctx, ds, cid, pi, split, seed, subset, n_t, tr, te, K, [0.0], set(), PREFILTER, False)
        assert len(e_rows) == len(m_rows) == len(K)
        for e, m in zip(e_rows, m_rows):
            assert e["k"] == m["k"] and m["gamma"] == 0.0 and m["method"] == "mvp"
            for col in ("auroc", "bacc", "mcc", "logloss", "brier"):
                assert abs(e[col] - m[col]) < 1e-12, (split, subset, n_t, m["k"], col)
            for col in ("n_selected", "exact_k", "n_filled", "C"):
                assert e[col] == m[col]
            compared += 1
        assert [(g["k"], g["rank"], g["gene"]) for g in e_genes] == [(g["k"], g["rank"], g["gene"]) for g in m_genes]
    assert compared == len(runs) * len(K)


def test_gamma_positive_selects_inside_source_prefilter(data_root):
    """γ > 0 lọc trên nguồn (Bước 1): mọi gen chọn ra phải nằm trong ctx.pre, kể cả khi L_T có gen khác phương sai lớn."""
    mvp = _load("run_rq2_mvp")
    ds, ctx, cid, pi, _, runs = _setup(data_root)
    split, seed, subset, n_t, tr, te = runs[0]
    _, genes, _ = mvp.run_one(ctx, ds, cid, pi, split, seed, subset, n_t, tr, te, K, [1.0], set(), PREFILTER, False)
    allowed = set(ctx.genes[ctx.pre].tolist())
    assert genes and all(g["gene"] in allowed and np.isfinite(g["pi_source"]) for g in genes)


def test_main_writes_outputs_and_resumes(data_root):
    mvp = _load("run_rq2_mvp")
    argv = ["--datasets", COHORT, "--k", *map(str, K), "--gamma", "0", "1", "--n-t", "5", "--prefilter", str(PREFILTER), "--save-scores",
            "--gamma-rule", "fixed", "--out", "RQ2_fixed"]
    mvp.main(argv)
    out = data_root / "results" / "survey_v2" / "RQ2_fixed" / COHORT
    per_run = pd.read_csv(out / "rq2_mvp_per_run.csv", dtype={"n_t": str})
    expected = ["cohort", "split", "seed", "subset", "n_t", "k", "method", "gamma", "gamma_rule", "auroc", "bacc", "mcc", "logloss", "brier",
                "C", "threshold", "n_selected", "exact_k", "n_filled", "n_ihc_in_panel", "select_time_s"]
    assert list(per_run.columns) == expected
    assert len(per_run) == 20 * len(K) * 2 and per_run.groupby(["split", "subset", "n_t", "gamma", "k"]).size().max() == 1
    assert set(per_run.gamma_rule) == {"fixed"} and (per_run.n_selected == per_run.k).all()
    genes = pd.read_csv(out / "rq2_mvp_selected_genes.csv", dtype={"n_t": str})
    assert list(genes.columns) == ["cohort", "split", "seed", "subset", "n_t", "k", "gamma", "rank", "gene", "pi_source"]
    assert len(genes) == per_run.k.sum()
    pi = pd.read_csv(out / "rq2_source_pi.csv")
    assert list(pi.columns) == ["gene", "pi", "C_q"] and len(pi) == PREFILTER and pi.pi.between(0, 1).all()
    scores = pd.read_csv(out / "rq2_mvp_scores.csv", dtype={"n_t": str})
    assert {"sample_id", "y_true", "score", "gamma", "k"} <= set(scores.columns)
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["experiment"] == "RQ2_mvp" and cfg["source_pi_sha256"] and cfg["fewshot_splits_sha256"] and cfg["gamma_rule"] == "fixed"
    mvp.main(argv)                                            # chạy lại: resume, không thêm dòng
    assert len(pd.read_csv(out / "rq2_mvp_per_run.csv")) == len(per_run)


# --- Bước 5: chọn γ theo M5, lồng trong L_T ---------------------------------------------------------------------------
def test_pick_gamma_prefers_smallest_unless_margin_reached():
    from survey_v2.mvp import pick_gamma

    assert pick_gamma({0.0: 0.70, 0.5: 0.695, 1.0: 0.692}) == 0.0      # giảm < 0,01 so với γ đang giữ: không đổi
    assert pick_gamma({0.0: 0.70, 0.5: 0.69}) == 0.5                   # giảm đúng 0,01: γ lớn hơn thắng
    assert pick_gamma({0.0: 0.70, 0.5: 0.60, 1.0: 0.595, 2.0: 0.50}) == 2.0
    assert pick_gamma({2.0: 0.50, 0.0: 0.70}) == 2.0                   # không phụ thuộc thứ tự khoá


@pytest.mark.parametrize("n_per_class, n_folds", [(5, 5), (10, 5), (20, 15), (40, 15)])
def test_inner_splits_follow_m5(n_per_class, n_folds):
    """n_T ≤ 10: 5 fold một lần; n_T ≥ 20: 5 fold lặp 3 lần. Mỗi lần lặp phủ mọi mẫu đúng một lần, có đủ hai lớp."""
    from survey_v2.mvp import inner_splits

    y = np.repeat([0, 1], n_per_class)
    splits = inner_splits(y, seed=3)
    assert len(splits) == n_folds
    n_repeats = n_folds // 5
    for r in range(n_repeats):
        val = np.concatenate([va for _, va in splits[r * 5:(r + 1) * 5]])
        assert sorted(val.tolist()) == list(range(len(y)))
    assert all(len(set(y[tr])) == 2 and len(set(y[va])) == 2 for tr, va in splits)
    assert [va.tolist() for _, va in inner_splits(y, seed=3)] == [va.tolist() for _, va in splits]


def test_gamma_path_reselects_genes_inside_each_fold():
    """Trên nhiễu thuần, chọn gen lại trong fold cho AUROC OOF gần 0,5; chọn một lần trên cả L_T (đối chứng) phải cao hơn hẳn."""
    from sklearn.metrics import roc_auc_score
    from survey_v2.classify import evaluate_panel
    from survey_v2.mvp import gamma_path, inner_splits, select_weighted

    honest, leaky = [], []
    for s in range(6):
        rng = np.random.default_rng(100 + s)
        X = rng.normal(size=(20, 300))
        y = np.repeat([0, 1], 10)
        path = gamma_path(X, y, np.zeros(300), np.arange(300), [0.0], [5], prefilter=300, seed=s)
        honest.append(path[(0.0, 5)]["oof_auroc"])
        panel = select_weighted(X, y, np.zeros(300), 0.0, [5], seed=s).panels[5]
        oof = np.zeros(20)
        for tr, va in inner_splits(y, seed=s):
            oof[va] = evaluate_panel(X[tr][:, panel], y[tr], X[va][:, panel], y[va], seed=s, small_n=True).scores_test
        leaky.append(roc_auc_score(y, oof))
    assert np.mean(honest) < 0.70
    assert np.mean(leaky) > np.mean(honest) + 0.20


def test_gamma0_prefilters_on_inner_train_only(monkeypatch):
    """γ = 0 lọc trên phần train của từng fold trong (như E1b trên L_T); γ > 0 giữ bộ lọc trên nguồn."""
    from survey_v2 import mvp as M

    seen = []
    real = M.variance_prefilter
    monkeypatch.setattr(M, "variance_prefilter", lambda X, p: (seen.append(X.shape[0]), real(X, p))[1])
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(20, 80)), np.repeat([0, 1], 10)
    M.gamma_path(X, y, np.zeros(30), np.arange(30), [1.0], [2], prefilter=30, seed=0)
    assert seen == []
    M.gamma_path(X, y, np.zeros(30), np.arange(30), [0.0], [2], prefilter=30, seed=0)
    assert seen == [16] * 5


def test_e_gamma_choice_ignores_test_labels(data_root):
    """Test (e): xáo nhãn test_T không đổi đường γ; chỉ số test thì đổi, chứng tỏ phép xáo có tác dụng."""
    mvp = _load("run_rq2_mvp")
    ds, ctx, cid, pi, _, runs = _setup(data_root)
    split, seed, subset, n_t, tr, te = runs[0]
    call = lambda: mvp.run_job(ctx, ds, cid, pi, split, seed, subset, n_t, tr, te, K, [0.0, 1.0], set(), set(), PREFILTER, False, "m5_cv")
    rows, _, _, path = call()
    index = {s: i for i, s in enumerate(ds.sample_ids)}
    te_idx = np.array([index[s] for s in te])
    ctx.yt = ctx.yt.copy()
    ctx.yt[te_idx] = 1 - ctx.yt[te_idx]
    rows2, _, _, path2 = call()
    assert path == path2 and path
    assert [r["auroc"] for r in rows] != [r["auroc"] for r in rows2]


def test_main_gamma_cv_outputs_and_resumes(data_root):
    mvp = _load("run_rq2_mvp")
    base = ["--datasets", COHORT, "--k", *map(str, K), "--gamma", "0", "1", "--n-t", "5", "--prefilter", str(PREFILTER), "--out", "RQ2_cv"]
    mvp.main([*base, "--gamma-rule", "fixed"])                     # fixed trước, m5_cv sau: không được nhân đôi dòng fixed
    mvp.main(base)
    out = data_root / "results" / "survey_v2" / "RQ2_cv" / COHORT
    per_run = pd.read_csv(out / "rq2_mvp_per_run.csv", dtype={"n_t": str})
    fixed, cv = per_run[per_run.gamma_rule == "fixed"], per_run[per_run.gamma_rule == "m5_cv"]
    assert len(fixed) == 20 * len(K) * 2 and len(cv) == 20 * len(K)
    assert fixed.groupby(["split", "subset", "n_t", "gamma", "k"]).size().max() == 1
    assert cv.groupby(["split", "subset", "n_t", "k"]).size().max() == 1
    path = pd.read_csv(out / "rq2_mvp_gamma_path.csv", dtype={"n_t": str})
    assert list(path.columns) == ["cohort", "split", "seed", "subset", "n_t", "k", "gamma", "oof_logloss", "oof_brier", "oof_auroc",
                                  "n_folds", "n_repeats", "selected"]
    assert len(path) == 20 * len(K) * 2 and path.groupby(["split", "subset", "n_t", "k"]).selected.sum().eq(1).all()
    picked = path[path.selected].merge(cv, on=["split", "subset", "n_t", "k"], suffixes=("_path", ""))
    assert (picked.gamma_path == picked.gamma).all() and len(picked) == len(cv)
    same = cv.merge(fixed, on=["split", "subset", "n_t", "k", "gamma"], suffixes=("", "_fixed"))
    assert len(same) == len(cv) and (same.auroc == same.auroc_fixed).all()   # dòng m5_cv là chỉ số của đúng γ đã chọn
    assert (same.select_time_s > same.select_time_s_fixed).all()             # có tính cả thời gian CV chọn γ
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["gamma_rule"] == "m5_cv"
    mvp.main(base)                                                 # chạy lại: resume, không thêm dòng
    assert len(pd.read_csv(out / "rq2_mvp_per_run.csv")) == len(per_run)
    assert len(pd.read_csv(out / "rq2_mvp_gamma_path.csv")) == len(path)


# --- Bước 7: test (b), (c), (d) ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_all(tmp_path_factory):
    """survey_v2.synthetic.make_all: nguồn và đích có chung gen tín hiệu; trả về (gốc, tên gen tín hiệu)."""
    from survey_v2 import config as C, synthetic

    root = tmp_path_factory.mktemp("kltn_mvp_all")
    paths = synthetic.make_all(root, n=160, p=400, effect=0.9, n_target=90)
    return root, set(json.loads((paths[C.SOURCE_DATASET] / "perm_config.json").read_text(encoding="utf-8"))["informative_genes"])


def _run_inputs(root, k_list):
    from survey_v2 import config as C
    from survey_v2.data import load_dataset
    from survey_v2.fewshot import build_fewshot_manifest, fewshot_runs
    from survey_v2.mvp import source_stability
    from survey_v2.transfer import TransferContext

    source, ds = load_dataset(root, C.SOURCE_DATASET, gene_space=None), load_dataset(root, COHORT)
    cid = C.cohort_id(COHORT)
    ctx = TransferContext(source, ds, k_list, selector="l1_lr", prefilter=PREFILTER, seed=C.seed_for(cid, 0, 0, 7))
    pi, _ = source_stability(ctx, cid)
    runs = [r for r in fewshot_runs(build_fewshot_manifest(COHORT, ds.sample_ids, ds.groups, ds.y)) if r[3] == "5"]
    return ds, ctx, cid, pi, runs


def test_b_source_weights_recover_more_signal_genes_at_small_n(synthetic_all, data_root):
    """Test (b): n_T = 5, nguồn có gen tín hiệu trùng đích: γ = 1 chọn trung bình nhiều gen tín hiệu hơn γ = 0."""
    mvp = _load("run_rq2_mvp")
    root, signal = synthetic_all
    ds, ctx, cid, pi, runs = _run_inputs(root, [5])
    assert signal & set(ctx.genes[ctx.pre].tolist())
    hits = {0.0: [], 1.0: []}
    for split, seed, subset, n_t, tr, te in runs:
        _, genes, _ = mvp.run_one(ctx, ds, cid, pi, split, seed, subset, n_t, tr, te, [5], [0.0, 1.0], set(), PREFILTER, False)
        for gamma in hits:
            hits[gamma].append(sum(g["gene"] in signal for g in genes if g["gamma"] == gamma))
    assert np.mean(hits[1.0]) > np.mean(hits[0.0])


def test_c_saturation_fills_with_ftest_when_k_exceeds_l1_capacity(synthetic_all, data_root):
    """Test (c): 2·n_T = k = 10: L1 trên 10 mẫu không đủ 10 hệ số, panel vẫn đúng k gen và n_filled được ghi."""
    mvp = _load("run_rq2_mvp")
    root, _ = synthetic_all
    ds, ctx, cid, pi, runs = _run_inputs(root, [10])
    rows, genes = [], []
    for split, seed, subset, n_t, tr, te in runs[:4]:
        r, g, _ = mvp.run_one(ctx, ds, cid, pi, split, seed, subset, n_t, tr, te, [10], [0.0, 1.0], set(), PREFILTER, False)
        rows += r
        genes += g
    assert all(r["n_selected"] == 10 for r in rows)
    assert any(r["n_filled"] > 0 for r in rows)
    assert all(not r["exact_k"] for r in rows if r["n_filled"] > 0)     # chiều ngược không đúng: M6 cắt còn k gen thì n_filled = 0
    panels = pd.DataFrame(genes).groupby(["split", "subset", "gamma"]).gene.agg(["size", "nunique"])
    assert (panels["size"] == 10).all() and (panels["nunique"] == 10).all()


@pytest.mark.slow
def test_d_two_hash_seeds_give_identical_outputs(data_root):
    """Test (d): hai PYTHONHASHSEED cho tệp trùng từng byte. Riêng rq2_mvp_per_run.csv bỏ cột thời gian select_time_s."""
    import os
    import subprocess

    outs = []
    for hash_seed in ("0", "1"):
        name = f"RQ2_hash{hash_seed}"
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "KLTN_DATA_ROOT": str(data_root), "KLTN_ALLOW_UNPUSHED": "1",
               "KLTN_ALLOW_UNTRACKED": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
        subprocess.run([sys.executable, str(REPO / "scripts" / "run_rq2_mvp.py"), "--datasets", COHORT, "--k", "2", "5", "--gamma", "0", "1",
                        "--n-t", "5", "--prefilter", str(PREFILTER), "--save-scores", "--out", name], check=True, cwd=REPO, env=env,
                       capture_output=True)
        outs.append(data_root / "results" / "survey_v2" / name / COHORT)
    for fname in ("rq2_mvp_selected_genes.csv", "rq2_mvp_gamma_path.csv", "rq2_mvp_scores.csv", "rq2_source_pi.csv"):
        assert (outs[0] / fname).read_bytes() == (outs[1] / fname).read_bytes(), fname
    a, b = (pd.read_csv(o / "rq2_mvp_per_run.csv", dtype=str).drop(columns="select_time_s") for o in outs)
    pd.testing.assert_frame_equal(a, b)


# --- Bước 8: thử nghiệm sớm trên TCGA ---------------------------------------------------------------------------------
def test_early_split_keeps_patients_apart_and_stratifies(data_root):
    from survey_v2 import config as C
    from survey_v2.data import load_dataset
    from survey_v2.mvp import early_split

    ds = load_dataset(data_root, C.SOURCE_DATASET, gene_space=None)
    src, tgt = early_split(ds)
    assert src.n + tgt.n == ds.n and abs(tgt.n / ds.n - 0.30) < 0.02
    assert not set(src.groups) & set(tgt.groups)
    assert abs(tgt.y.mean() - ds.y.mean()) < 0.03
    assert src.name != ds.name and tgt.name != ds.name and src.X.shape[1] == ds.p
    assert np.array_equal(early_split(ds)[1].sample_ids, tgt.sample_ids)
    j = 3
    assert tgt.X[j, 0] == ds.X[list(ds.sample_ids).index(tgt.sample_ids[j]), 0]


def test_main_early_tcga_compares_three_arms_without_target_leak(data_root):
    from survey_v2 import config as C
    from survey_v2.data import load_dataset
    from survey_v2.mvp import early_split, source_stability
    from survey_v2.transfer import TransferContext

    mvp = _load("run_rq2_mvp")
    argv = ["--early-tcga", "--n-t", "5", "10", "--k", "2", "5", "--prefilter", str(PREFILTER), "--out", "RQ2_early"]
    mvp.main(argv)
    out = data_root / "results" / "survey_v2" / "RQ2_early" / "tcga_early"
    split = pd.read_csv(out / "early_split.csv")
    assert list(split.columns) == ["sample_id", "patient_id", "label", "role"] and set(split.role) == {"source", "target"}
    assert not set(split[split.role == "source"].patient_id) & set(split[split.role == "target"].patient_id)
    man = pd.read_csv(out / "fewshot_splits.csv", dtype={"n_t": str})
    assert set(man.n_t) == {"5", "10"} and set(man.sample_id) <= set(split[split.role == "target"].sample_id)

    # π chỉ từ nguồn giả: tính lại độc lập trên nguồn giả phải trùng tệp đã ghi
    src, tgt = early_split(load_dataset(data_root, C.SOURCE_DATASET, gene_space=None))
    cid = C.cohort_id(tgt.name)
    ctx = TransferContext(src, tgt, [2, 5], selector="l1_lr", prefilter=PREFILTER, seed=C.seed_for(cid, 0, 0, 7))
    assert np.array_equal(source_stability(ctx, cid)[0], pd.read_csv(out / "rq2_source_pi.csv").pi.to_numpy())

    per_run = pd.read_csv(out / "rq2_mvp_per_run.csv", dtype={"n_t": str})
    sos = pd.read_csv(out / "rq2_early_select_on_source.csv", dtype={"n_t": str})
    assert len(sos) == 40 * 2 and set(sos.method) == {"select_on_source"} and (sos.n_selected == sos.k).all()
    summary = pd.read_csv(out / "rq2_early_summary.csv", dtype={"n_t": str})
    assert list(summary.columns) == ["n_t", "k", "arm", "n_runs", "auroc_mean", "auroc_sd", "bacc_mean", "bacc_sd", "logloss_mean", "logloss_sd",
                                     "ihc_panel_rate"]
    assert len(summary) == 2 * 2 * 3 and set(summary.arm) == {"mvp_m5", "target_only", "select_on_source"} and (summary.n_runs == 20).all()
    assert summary.ihc_panel_rate.between(0, 1).all()
    pick = lambda arm: summary[(summary.arm == arm) & (summary.n_t == "5") & (summary.k == 5)].iloc[0]
    rows = per_run[(per_run.n_t == "5") & (per_run.k == 5)]
    target = rows[(rows.gamma_rule == "fixed") & (rows.gamma == 0)]
    assert pick("target_only").auroc_mean == pytest.approx(target.auroc.mean())
    assert pick("target_only").auroc_sd == pytest.approx(target.auroc.std())
    assert pick("target_only").ihc_panel_rate == pytest.approx((target.n_ihc_in_panel > 0).mean())
    assert pick("mvp_m5").auroc_mean == pytest.approx(rows[rows.gamma_rule == "m5_cv"].auroc.mean())
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["experiment"] == "RQ2_mvp_early" and cfg["early_seed"] == 42 and cfg["n_source"] == int((split.role == "source").sum())
    assert cfg["early_split_sha256"] and cfg["fewshot_splits_sha256"]
    mvp.main(argv)                                                 # chạy lại: resume, không thêm dòng
    assert len(pd.read_csv(out / "rq2_mvp_per_run.csv")) == len(per_run)
    assert len(pd.read_csv(out / "rq2_early_select_on_source.csv")) == len(sos)
