"""Kiểm tra các phương pháp chọn gen, wrapper, mRMR và lọc sơ bộ của survey_v2 (dữ liệu tổng hợp nhỏ)."""

from __future__ import annotations

import numpy as np
import pytest

from survey_v2 import config as C
from survey_v2.data import add_correlated_copies, stratified_subsample, variance_prefilter
from survey_v2.methods import prefix_panels, select
from survey_v2.mrmr import discretize, mrmr_ranking, relevance
from survey_v2.wrappers import InnerCVFitness, binary_pso, genetic_algorithm


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n, p = 90, 120
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(0, 1, size=(n, p))
    X[:, :6] += 1.5 * y[:, None]           # 6 gen tín hiệu
    X[:, 6] = X[:, 0] * 0.95 + 0.3 * rng.normal(size=n)   # gen dư thừa với gen 0
    return X, y


def test_variance_prefilter_is_label_free_and_sorted(toy):
    X, _ = toy
    keep = variance_prefilter(X, 50)
    assert len(keep) == 50 and np.all(np.diff(keep) > 0)
    var = X.var(axis=0, ddof=1)
    assert set(keep) == set(np.argsort(-var)[:50])
    assert len(variance_prefilter(X, 10_000)) == X.shape[1]


def test_prefix_panels_are_nested():
    order = np.arange(100)[::-1]
    panels = prefix_panels(order, [5, 10, 20])
    assert set(panels[5]) <= set(panels[10]) <= set(panels[20])
    with pytest.raises(ValueError):
        prefix_panels(order, [200])


@pytest.mark.parametrize("method", ["f_test", "mutual_info", "mrmr", "relieff", "rf_importance", "svm_rfe", "l1_lr", "elastic_net"])
def test_select_returns_exact_k_and_nested_for_rankers(toy, method):
    X, y = toy
    ks = [5, 10, 20]
    sel = select(method, X, y, ks, seed=1)
    for k in ks:
        assert len(sel.panels[k]) == k
        assert len(set(sel.panels[k].tolist())) == k
        assert sel.panels[k].min() >= 0 and sel.panels[k].max() < X.shape[1]
    if method in C.RANKING_METHODS:
        assert set(sel.panels[5]) <= set(sel.panels[10]) <= set(sel.panels[20])
    if method in ("f_test", "mrmr", "svm_rfe", "l1_lr"):
        # phần lớn gen tín hiệu phải nằm trong top 10
        assert len(set(sel.panels[10]) & set(range(7))) >= 4


def test_mrmr_penalises_redundant_gene(toy):
    X, y = toy
    D = discretize(X)
    assert set(np.unique(D)) <= {0, 1, 2}
    rel = relevance(D, y)
    assert rel[:6].mean() > rel[20:].mean()
    order = mrmr_ranking(X, y, 10, criterion="MID")
    assert len(order) == 10 and len(set(order.tolist())) == 10
    # gen 6 gần như trùng gen 0: không được đứng liền ngay sau gen 0 ở vị trí thứ hai trong mRMR
    pos0, pos6 = list(order).index(0) if 0 in order else -1, list(order).index(6) if 6 in order else 99
    assert not (pos0 == 0 and pos6 == 1)
    order_q = mrmr_ranking(X, y, 10, criterion="MIQ")
    assert len(order_q) == 10


def test_sparse_exact_k_reports_fill_when_saturated():
    rng = np.random.default_rng(3)
    n, p = 30, 60
    y = np.array([0, 1] * 15)
    X = rng.normal(size=(n, p))
    X[:, 0] += 2 * y
    sel = select("l1_lr", X, y, [10, 50], seed=0)
    assert len(sel.panels[50]) == 50
    # L1 với n = 30 không thể có 50 hệ số khác 0 -> phải bổ sung bằng F-test và ghi n_filled > 0
    assert sel.info["n_filled"][50] > 0
    assert sel.info["exact"][50] is False


def test_wrappers_return_exact_k_and_count_evaluations(toy):
    X, y = toy
    fit = InnerCVFitness(X, y, seed=0)
    rng = np.random.default_rng(0)
    best, f, hist = genetic_algorithm(fit, X.shape[1], 5, rng, pop=6, generations=3)
    assert len(best) == 5 and len(set(best.tolist())) == 5 and 0 <= f <= 1 and len(hist) == 4
    assert fit.n_eval > 0 and fit.n_eval <= 6 + 6 * 3
    fit2 = InnerCVFitness(X, y, seed=0)
    best2, f2, hist2 = binary_pso(fit2, X.shape[1], 5, np.random.default_rng(0), pop=6, iterations=3)
    assert len(best2) == 5 and len(set(best2.tolist())) == 5
    assert all(b >= a for a, b in zip(hist2, hist2[1:]))  # gbest không giảm
    sel = select("ga", X, y, [5], seed=0, wrapper_kwargs={"pop": 6, "generations": 2})
    assert len(sel.panels[5]) == 5 and sel.info["n_eval"][5] > 0


def test_fitness_uses_only_training_rows(toy):
    """Fitness được dựng chỉ từ (X_tr, y_tr): thay đổi dữ liệu ngoài không ảnh hưởng."""
    X, y = toy
    Xtr, ytr = X[:60], y[:60]
    f1 = InnerCVFitness(Xtr, ytr, seed=0)([0, 1, 2])
    X_mod = X.copy()
    X_mod[60:] += 100.0
    f2 = InnerCVFitness(X_mod[:60], ytr, seed=0)([0, 1, 2])
    assert f1 == f2


def test_subsample_and_copies(toy):
    X, y = toy
    rng = np.random.default_rng(0)
    idx = stratified_subsample(y, 0.5, rng)
    assert len(idx) == pytest.approx(len(y) * 0.5, abs=2)
    assert abs(y[idx].mean() - y.mean()) < 0.1
    genes = np.array([f"g{j}" for j in range(X.shape[1])])
    X2, genes2, src, copies = add_correlated_copies(X, genes, 0.9, 5, rng)
    assert X2.shape == (X.shape[0], X.shape[1] + 5) and len(genes2) == X.shape[1] + 5
    for s, c in zip(src, copies):
        r = np.corrcoef(X2[:, s], X2[:, c])[0, 1]
        assert r > 0.75
        assert genes2[c].startswith(genes[s] + "__copy_rho")


# ---------------------------------------------------------------------------------------------------
# v3.1.6 (Review 31/10/2026): tái lập SVM-RFE (QĐ25), abess với cột hằng (QĐ26), bộ nhớ tiến trình con (QĐ27)
# ---------------------------------------------------------------------------------------------------
def test_svm_rfe_is_reproducible_with_seed_when_p_gt_n():
    """p > n -> LinearSVC(dual='auto') dùng bộ giải dual (ngẫu nhiên); random_state = seed phải cho cùng xếp hạng."""
    rng = np.random.default_rng(3)
    n, p = 40, 300
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(n, p))
    X[:, :4] += 0.8 * y[:, None]
    a = select("svm_rfe", X, y, [5, 20], seed=7)
    b = select("svm_rfe", X, y, [5, 20], seed=7)
    assert a.panels[20].tolist() == b.panels[20].tolist() and a.ranking.tolist() == b.ranking.tolist()
    c = select("svm_rfe", X, y, [5, 20], seed=8)
    assert len(c.panels[20]) == 20                       # seed khác vẫn hợp lệ (có thể khác xếp hạng)


def test_abess_drops_constant_genes_and_maps_indices_back():
    pytest.importorskip("abess")
    rng = np.random.default_rng(5)
    n, p = 80, 60
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(n, p))
    X[:, :3] += 1.5 * y[:, None]
    const_cols = [4, 10, 59]
    X[:, const_cols] = 7.0                               # cột hằng (sd = 0): abess 0.4.11 báo OverflowError nếu đưa vào
    sel = select("abess", X, y, [2, 5], seed=0)
    assert sel.info["n_constant_dropped"] == 3
    for k in (2, 5):
        assert len(sel.panels[k]) == k and not (set(sel.panels[k]) & set(const_cols))
        assert sel.panels[k].max() < p
    assert len(set(sel.panels[5]) & {0, 1, 2}) >= 2        # gen tín hiệu vẫn được chọn (chỉ số đã ánh xạ đúng)
    plain = select("abess", X[:, [j for j in range(p) if j not in const_cols]], y, [5], seed=0)
    kept = np.array([j for j in range(p) if j not in const_cols])
    assert sorted(kept[plain.panels[5]].tolist()) == sorted(sel.panels[5].tolist())   # giống kết quả trên ma trận đã bỏ cột hằng


def test_child_peak_rss_does_not_inherit_parent_peak():
    """QĐ27: ru_maxrss của tiến trình con kế thừa đỉnh RSS của cha (Linux); peak_rss_mb() phải là của chính tiến trình con."""
    import json
    import subprocess
    import sys
    from survey_v2.timing import peak_rss_mb
    big = np.ones((64, 1_000_000))                      # ~ 512 MB trong tiến trình cha
    parent_peak = peak_rss_mb()
    assert parent_peak > 400 and big.shape[0] == 64
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); from survey_v2.timing import peak_rss_mb, _maxrss_mb; "
            "print(json.dumps({'peak': peak_rss_mb(), 'ru': _maxrss_mb()}))")
    import survey_v2
    repo = str(__import__("pathlib").Path(survey_v2.__file__).resolve().parents[1])
    res = subprocess.run([sys.executable, "-c", code, repo], capture_output=True, text=True, check=True)
    child = json.loads(res.stdout.strip().splitlines()[-1])
    assert child["peak"] < 200, child                   # tiến trình con chỉ nạp numpy: vài chục MB
    if sys.platform.startswith("linux"):
        assert child["ru"] >= parent_peak * 0.9, child  # minh hoạ lỗi cũ: ru_maxrss của con ≈ đỉnh của cha


def test_measure_reports_time_without_tracemalloc(toy):
    from survey_v2.timing import measure
    X, y = toy
    calls = []

    def sel(Xs, ys, k):
        calls.append(Xs.shape)
        return np.arange(k)

    r = measure(sel, X, y, 5, repeats=2)
    assert r["repeats"] == 2 and len(r["wall_s_all"]) == 2 and r["wall_s"] >= 0
    assert "peak_mb" not in r and "maxrss_mb" not in r
    assert calls[0][0] <= 40 and calls[0][0] < X.shape[0] and calls[1] == X.shape   # khởi động trên lát nhỏ rồi đo trên dữ liệu đủ
