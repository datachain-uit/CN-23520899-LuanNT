"""Mười ba phương pháp chọn gen + tham chiếu của survey v3 (QĐ15, Review 26/10/2026; KBTN v2 Mục 5.1).

So với v2: thêm abess (L0, ràng buộc cứng |S| = k), stability selection (Meinshausen & Bühlmann 2010) và k-TSP
(Tan et al. 2005); SVM-RFE xếp hạng tới k_min = 1; panel cố định (IHC, panel công bố) ở survey_v2/panels.py.

Quy ước chung
-------------
* Mọi hàm chỉ nhận PHẦN TRAIN của fold (đã lọc sơ bộ p0 gen). Không hàm nào nhận phần test.
* Phương pháp xếp hạng (RANKING_METHODS) trả về một thứ tự đầy đủ/đủ dài; tập k gen = k gen đầu
  (tiền tố), nên các tập lồng nhau theo k (assert trong `prefix_panels`).
* Phương pháp theo k (l1_lr, elastic_net, abess) trả về đúng k gen cho từng k; k-TSP chỉ ở k chẵn (panel[k] = None ở k lẻ); khi L1 không đạt đủ k gen
  (bão hòa ≤ n_train), bổ sung theo thứ hạng F-test trên train và ghi `n_filled` (KBTN 5.1).
* Chuẩn hóa z-score (fit trên train) được áp dụng TRƯỚC l1_lr, elastic_net, svm_rfe (M7).
* Chỉ số trả về là chỉ số cột của ma trận đưa vào (ma trận đã lọc sơ bộ); người gọi ánh xạ về gen.
* v3.1.6 (QĐ25, 31/10/2026): mọi bộ giải có ngẫu nhiên nhận `random_state = seed` — SVM-RFE dùng LinearSVC dual
  (liblinear xáo trộn toạ độ) nên trước đây không tái lập giữa hai phiên; abess bỏ cột hằng (sd = 0) trước khi khớp
  (OverflowError ở p = toàn bộ gen TCGA, QĐ26) và ánh xạ chỉ số về ma trận gốc.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from . import config as C
from .mrmr import mrmr_ranking


# ---------------------------------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------------------------------
def zscore_fit(X: np.ndarray):
    mu = X.mean(0)
    sd = X.std(0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def ftest_order(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    F, _ = f_classif(X, y)
    return np.argsort(-np.nan_to_num(F, nan=-np.inf), kind="stable")


@dataclass
class Selection:
    """Kết quả chọn gen của một phương pháp trên một phần train."""
    method: str
    panels: dict                       # k -> np.ndarray chỉ số (kích thước đúng k)
    ranking: np.ndarray | None = None  # thứ tự đầy đủ (phương pháp xếp hạng)
    info: dict = field(default_factory=dict)   # C đã dùng, n_filled, n_eval, ...


def prefix_panels(order: np.ndarray, k_list) -> dict:
    """Tập k gen = k gen đầu của xếp hạng; kiểm tra lồng nhau."""
    ks = sorted(int(k) for k in k_list)
    if max(ks) > len(order):
        raise ValueError(f"xếp hạng chỉ có {len(order)} gen, cần k_max = {max(ks)}")
    panels = {k: np.asarray(order[:k], dtype=int) for k in ks}
    for a, b in zip(ks, ks[1:]):
        assert set(panels[a]) <= set(panels[b])
    return panels


# ---------------------------------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------------------------------
def rank_f_test(X, y, **_) -> np.ndarray:
    """ANOVA F-test (tương đương t-test hai lớp): O(n p)."""
    return ftest_order(X, y)


def rank_mutual_info(X, y, seed: int = 0, n_neighbors: int = 3, **_) -> np.ndarray:
    """Thông tin tương hỗ (ước lượng KNN của scikit-learn): O(n log n · p)."""
    mi = mutual_info_classif(X, y, n_neighbors=n_neighbors, random_state=seed)
    return np.argsort(-mi, kind="stable")


def rank_mrmr(X, y, k_max: int, **_) -> np.ndarray:
    """mRMR (Peng et al., 2005), MID/MIQ trên dữ liệu rời rạc hóa; chỉ xếp k_max gen đầu (tham lam)."""
    return mrmr_ranking(X, y, k_max, criterion=C.MRMR_CRITERION, discretize_method=C.MRMR_DISCRETIZE)


def rank_relieff(X, y, n_neighbors: int = C.RELIEFF_NEIGHBORS, n_jobs: int = 1, **_) -> np.ndarray:
    """ReliefF (Kononenko 1994; cài đặt skrebate, Urbanowicz et al. 2018): O(n² p)."""
    from skrebate import ReliefF  # import muộn để module vẫn dùng được khi thiếu skrebate
    est = ReliefF(n_features_to_select=X.shape[1], n_neighbors=n_neighbors, n_jobs=n_jobs)
    est.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=int))
    return np.argsort(-est.feature_importances_, kind="stable")


# ---------------------------------------------------------------------------------------------------
# Embedded
# ---------------------------------------------------------------------------------------------------
def rank_rf_importance(X, y, seed: int = 0, n_trees: int = C.RF_N_TREES, n_jobs: int = 1, **_) -> np.ndarray:
    """Độ quan trọng Gini của Random Forest (Breiman 2001): O(B n p log n)."""
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=n_jobs,
                                class_weight="balanced_subsample")
    rf.fit(X, y)
    return np.argsort(-rf.feature_importances_, kind="stable")


def rank_svm_rfe(X, y, k_min: int, step: float = 0.1, C_svm: float = 1.0, seed: int = 0, **_) -> np.ndarray:
    """SVM-RFE (Guyon et al., 2002) với xếp hạng CHẶT: loại dần một tỉ lệ `step` gen có w² nhỏ nhất
    cho tới khi còn k_min gen; thứ tự loại (kèm w² trong mỗi lô) là xếp hạng ngược.

    X phải đã chuẩn hóa z-score trên train (M7). Trả về thứ tự đầy đủ, gen quan trọng nhất trước.
    `seed` -> random_state của LinearSVC: khi p > n, dual="auto" chọn bộ giải dual của liblinear (xáo trộn toạ độ
    ngẫu nhiên), không có random_state thì hai lần chạy cho hai xếp hạng khác nhau (QĐ25, v3.1.6).
    """
    p = X.shape[1]
    active = np.arange(p)
    removed_order: list[int] = []                      # ít quan trọng nhất trước
    while True:
        svm = LinearSVC(C=C_svm, class_weight="balanced", dual="auto", max_iter=10000, random_state=int(seed))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svm.fit(X[:, active], y)
        w2 = np.ravel(svm.coef_) ** 2
        order = np.argsort(w2, kind="stable")          # tăng dần: ít quan trọng trước
        if len(active) <= k_min:
            final = active[np.argsort(-w2, kind="stable")]  # quan trọng nhất trước
            break
        n_remove = max(1, int(step * len(active)))
        n_remove = min(n_remove, len(active) - k_min)
        drop = order[:n_remove]
        removed_order.extend(active[drop].tolist())
        active = np.sort(active[order[n_remove:]])
    ranking = np.concatenate([final, np.asarray(removed_order[::-1], dtype=int)])
    assert len(ranking) == p and len(set(ranking)) == p
    return ranking


def _coef_path(model_factory, X, y, c_grid):
    """Fit dọc lưới C (tăng dần); trả về danh sách (C, hệ số)."""
    out = []
    for c in c_grid:
        m = model_factory(c)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X, y)
        out.append((float(c), np.ravel(m.coef_).copy()))
    return out


def _exact_k_from_path(model_factory, X, y, k, path, refine_steps: int = 6):
    """Tìm C cho đúng k hệ số khác 0 (tinh chỉnh nhị phân giữa hai điểm lưới); nếu không hit đúng,
    lấy C nhỏ nhất có ≥ k hệ số khác 0 rồi giữ k gen |β| lớn nhất (M6)."""
    nnz = [int((coef != 0).sum()) for _, coef in path]
    exact = [i for i, v in enumerate(nnz) if v == k]
    if exact:
        c, coef = path[exact[0]]
        return np.flatnonzero(coef != 0), c, True
    above = [i for i, v in enumerate(nnz) if v > k]
    if not above:
        return None, None, False                          # bão hòa: không C nào đạt k
    i_hi = above[0]                                       # C nhỏ nhất có nnz > k
    c_hi, coef_hi = path[i_hi]
    c_lo = path[i_hi - 1][0] if i_hi > 0 else c_hi / 10.0
    best_c, best_coef = c_hi, coef_hi
    for _ in range(refine_steps):
        c_mid = float(np.sqrt(c_lo * c_hi))
        m = model_factory(c_mid)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X, y)
        coef = np.ravel(m.coef_)
        v = int((coef != 0).sum())
        if v == k:
            return np.flatnonzero(coef != 0), c_mid, True
        if v > k:
            c_hi, best_c, best_coef = c_mid, c_mid, coef
        else:
            c_lo = c_mid
    top = np.argsort(-np.abs(best_coef), kind="stable")[:k]
    return np.sort(top), best_c, False


def _fill_with_ftest(sel: np.ndarray, k: int, X, y) -> tuple[np.ndarray, int]:
    if len(sel) >= k:
        return sel[:k], 0
    order = ftest_order(X, y)
    extra = [j for j in order if j not in set(sel.tolist())][: k - len(sel)]
    return np.concatenate([sel, np.asarray(extra, dtype=int)]), len(extra)


def _l1_factory(seed):
    return lambda c: LogisticRegression(penalty="l1", solver="liblinear", C=c, class_weight="balanced",
                                        max_iter=5000, random_state=seed)


def _enet_factory(seed, l1_ratio=0.5):
    return lambda c: LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=l1_ratio, C=c,
                                        class_weight="balanced", max_iter=3000, tol=1e-3, random_state=seed)


def select_sparse_exact_k(X, y, k_list, kind: str, seed: int = 0,
                          c_grid=None) -> Selection:
    """L1-LR hoặc Elastic Net (logistic) với đúng k gen cho từng k (X đã z-score trên train).

    Đường hệ số được fit MỘT lần trên lưới C (log), dùng chung cho mọi k; sau đó tinh chỉnh nhị phân
    quanh điểm lưới để hit đúng k. Khi L1 bão hòa (nnz tối đa < k, thường vì k > n_train),
    bổ sung theo F-test và ghi n_filled (KBTN 5.1).
    """
    if c_grid is None:
        c_grid = np.logspace(-3, 2, 26)
    factory = _l1_factory(seed) if kind == "l1_lr" else _enet_factory(seed)
    path = _coef_path(factory, X, y, c_grid)
    panels, info = {}, {"C": {}, "exact": {}, "n_filled": {}}
    for k in sorted(int(k) for k in k_list):
        sel, c, exact = _exact_k_from_path(factory, X, y, k, path)
        if sel is None:
            # bão hòa: lấy nghiệm có nhiều hệ số khác 0 nhất rồi bổ sung
            c, coef = max(path, key=lambda t: int((t[1] != 0).sum()))
            sel = np.flatnonzero(coef != 0)
            sel = sel[np.argsort(-np.abs(coef[sel]), kind="stable")]
            exact = False
        sel, n_filled = _fill_with_ftest(np.asarray(sel, dtype=int), k, X, y)
        panels[k] = np.sort(sel[:k])
        info["C"][k], info["exact"][k], info["n_filled"][k] = c, bool(exact), n_filled
        assert len(panels[k]) == k and len(set(panels[k].tolist())) == k
    return Selection(method=kind, panels=panels, info=info)


# ---------------------------------------------------------------------------------------------------
# Ràng buộc cứng, tần suất chọn, thứ hạng cặp (KBTN v2 Mục 5.1)
# ---------------------------------------------------------------------------------------------------
def _stratified_half(y: np.ndarray, rng: np.random.Generator, fraction: float = C.STABSEL_FRACTION) -> np.ndarray:
    idx = []
    for c in (0, 1):
        cand = np.flatnonzero(y == c)
        m = max(2, int(round(fraction * len(cand))))
        idx.append(rng.choice(cand, size=min(m, len(cand)), replace=False))
    return np.sort(np.concatenate(idx))


def rank_stability_selection(X, y, k_max: int, seed: int = 0, n_subsamples: int = C.STABSEL_N_SUBSAMPLES,
                             fraction: float = C.STABSEL_FRACTION, **_) -> tuple[np.ndarray, dict]:
    """Stability selection (Meinshausen & Bühlmann 2010) với L1-LR (X đã z-score trên train).

    Mức phạt được chọn MỘT lần trên toàn bộ phần train sao cho số hệ số khác 0 ≈ q = STABSEL_TARGET_NNZ_FACTOR·k_max
    (q của M&B); sau đó fit L1-LR tại C đó trên `n_subsamples` tập con phân tầng 50 %; π_j = tần suất β_j ≠ 0.
    Xếp hạng theo π_j giảm dần, hoà theo |β| trung bình; trả về (thứ tự, {"C": C_q, "pi": π}).
    """
    n, p = X.shape
    q = int(min(C.STABSEL_TARGET_NNZ_FACTOR * k_max, max(2, n // 2 - 1), p))
    factory = _l1_factory(seed)
    path = _coef_path(factory, X, y, np.logspace(-3, 2, 26))
    sel, c_q, _ = _exact_k_from_path(factory, X, y, q, path, refine_steps=4)
    if c_q is None:                       # bão hoà: lấy C lớn nhất của lưới
        c_q = float(path[-1][0])
    rng = np.random.default_rng(seed)
    counts = np.zeros(p)
    abs_coef = np.zeros(p)
    for _ in range(n_subsamples):
        idx = _stratified_half(y, rng, fraction)
        m = factory(c_q)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X[idx], y[idx])
        coef = np.ravel(m.coef_)
        counts += coef != 0
        abs_coef += np.abs(coef)
    pi = counts / n_subsamples
    order = np.lexsort((-abs_coef, -pi))   # π giảm dần, hoà: |β| trung bình giảm dần
    return np.asarray(order, dtype=int), {"C": float(c_q), "pi": pi, "q_target": q}


def ktsp_pair_scores(X, y, max_p: int = C.KTSP_MAX_P, chunk: int = 16):
    """Điểm cặp gen của TSP (Geman et al. 2004; Tan et al. 2005): Δ_ij = |P(x_i < x_j | y = 0) − P(x_i < x_j | y = 1)|,
    tính chính xác cho mọi cặp i < j (O(n p^2), theo khối). Trả về (Δ, γ) với γ_ij = |mean rank diff| để phá hoà.
    """
    n, p = X.shape
    if p > max_p:
        raise ValueError(f"k-TSP: p = {p} > KTSP_MAX_P = {max_p}; lọc sơ bộ trước")
    X0, X1 = X[y == 0], X[y == 1]
    # thứ hạng trong mẫu (tiêu chí phụ của Tan 2005): trung bình rank theo lớp, tính một lần
    r0 = np.argsort(np.argsort(X0, axis=1), axis=1).mean(axis=0).astype(np.float32)   # (p,)
    r1 = np.argsort(np.argsort(X1, axis=1), axis=1).mean(axis=0).astype(np.float32)
    delta = np.zeros((p, p), dtype=np.float32)
    for start in range(0, p, chunk):
        sl = slice(start, min(start + chunk, p))
        p0 = (X0[:, sl, None] < X0[:, None, :]).mean(axis=0)   # (chunk, p): P(x_i < x_j | 0)
        p1 = (X1[:, sl, None] < X1[:, None, :]).mean(axis=0)
        delta[sl] = np.abs(p0 - p1)
    # γ_ij = |E[r_i − r_j | 0] − E[r_i − r_j | 1]| (tuyến tính theo trung bình rank nên không cần khối)
    dr = (r0 - r1)
    gamma = np.abs(dr[:, None] - dr[None, :]).astype(np.float32)
    iu = np.triu_indices(p, 1)
    delta_sym = np.maximum(delta, delta.T)
    gamma_sym = np.maximum(gamma, gamma.T)
    return delta_sym[iu], gamma_sym[iu], iu


def rank_ktsp(X, y, k_max: int, **_) -> tuple[np.ndarray, dict]:
    """k-TSP: chọn tham lam các cặp gen RỜI NHAU theo Δ giảm dần (hoà theo γ); panel k = 2m gen = m cặp đầu.

    Trả về thứ tự gen ghép theo cặp [i1, j1, i2, j2, ...] (tiền tố chẵn là panel hợp lệ) và danh sách cặp.
    Chỉ k chẵn là panel k-TSP đúng nghĩa; k lẻ được ghi NaN ở script (KBTN 5.1).
    """
    d, g, (ii, jj) = ktsp_pair_scores(X, y)
    order = np.lexsort((-g, -d))
    used = set()
    pairs = []
    need = max(1, k_max // 2)
    for t in order:
        a, b = int(ii[t]), int(jj[t])
        if a in used or b in used:
            continue
        pairs.append((a, b, float(d[t]), float(g[t])))
        used.update((a, b))
        if len(pairs) >= need:
            break
    ranking = np.array([x for a, b, _, _ in pairs for x in (a, b)], dtype=int)
    return ranking, {"pairs": pairs}


def select_abess(X, y, k_list, seed: int = 0) -> Selection:
    """L0 / best-subset logistic (abess; Zhu et al. 2020, 2022) với support_size = k cho từng k (X đã z-score).

    v3.1.6 (QĐ26): cột hằng (sd = 0 trên train; sau z-score là cột toàn 0) bị loại trước khi khớp — abess 0.4.11 báo
    OverflowError khi ma trận có cột hằng (gặp ở E3, p = toàn bộ 19.938 gen TCGA: 418 gen hằng trong fold 0). Chỉ số
    trả về vẫn là chỉ số cột của ma trận ĐƯA VÀO; `info["n_constant_dropped"]` ghi số cột đã loại. Ở p0 = 5.000 sau
    lọc phương sai không có cột hằng nên kết quả E1/E1b/E4 không đổi.
    """
    try:
        from abess import LogisticRegression as AbessLR
    except ImportError as exc:   # pragma: no cover
        raise ImportError("abess chưa cài: pip install abess (ghi 'không chạy' trong HANDOVER nếu không cài được)") from exc
    X = np.asarray(X)
    keep = np.flatnonzero(X.std(axis=0, ddof=0) > 0)
    n_constant = X.shape[1] - len(keep)
    Xk = X if n_constant == 0 else X[:, keep]
    panels, info = {}, {"exact": {}, "n_filled": {}, "n_constant_dropped": int(n_constant)}
    for k in sorted(int(k) for k in k_list):
        if k > Xk.shape[1]:
            raise ValueError(f"abess: k = {k} lớn hơn số gen không hằng ({Xk.shape[1]})")
        m = AbessLR(support_size=int(k))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xk, y)
        coef = np.ravel(m.coef_)
        sel = np.flatnonzero(coef != 0)
        sel = sel[np.argsort(-np.abs(coef[sel]), kind="stable")]
        exact = len(sel) == k
        if len(sel) > k:
            sel = sel[:k]
        sel, n_filled = _fill_with_ftest(np.asarray(sel, dtype=int), k, Xk, y)
        sel = sel[:k] if n_constant == 0 else keep[sel[:k]]        # ánh xạ về chỉ số của ma trận đưa vào
        panels[k] = np.sort(sel)
        info["exact"][k], info["n_filled"][k] = bool(exact), n_filled
        assert len(panels[k]) == k
    return Selection(method="abess", panels=panels, info=info)


def ktsp_vote_scores(X, pairs) -> np.ndarray:
    """Luật bỏ phiếu k-TSP (phụ): điểm = tỉ lệ cặp (i, j) có x_i < x_j (hướng cặp đã chuẩn hoá theo lớp 1)."""
    votes = np.zeros(X.shape[0])
    for a, b, _, _ in pairs:
        votes += (X[:, a] < X[:, b]).astype(float)
    return votes / max(1, len(pairs))


# ---------------------------------------------------------------------------------------------------
# Tham chiếu
# ---------------------------------------------------------------------------------------------------
def random_panel(p: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(p, size=k, replace=False))


# ---------------------------------------------------------------------------------------------------
# Giao diện chung
# ---------------------------------------------------------------------------------------------------
def select(method: str, X_tr: np.ndarray, y_tr: np.ndarray, k_list, seed: int = 0, n_jobs: int = 1,
           wrapper_kwargs: dict | None = None) -> Selection:
    """Chọn gen cho một phương pháp trên PHẦN TRAIN (đã lọc sơ bộ). Trả về Selection với panels theo k.

    Chuẩn hóa z-score (fit trên X_tr) được áp dụng bên trong cho các phương pháp ở STANDARDIZE_BEFORE.
    Wrapper (ga, bpso) dùng survey_v2.wrappers; fitness chỉ dùng X_tr/y_tr (CV nội bộ).
    """
    ks = sorted(int(k) for k in k_list)
    k_max, k_min = max(ks), min(ks)
    X_use = X_tr
    if method in C.STANDARDIZE_BEFORE:
        mu, sd = zscore_fit(X_tr)
        X_use = zscore_apply(X_tr, mu, sd)

    if method == "f_test":
        order = rank_f_test(X_use, y_tr)
    elif method == "mutual_info":
        order = rank_mutual_info(X_use, y_tr, seed=seed)
    elif method == "mrmr":
        order = rank_mrmr(X_use, y_tr, k_max=k_max)
    elif method == "relieff":
        order = rank_relieff(X_use, y_tr, n_jobs=n_jobs)
    elif method == "rf_importance":
        order = rank_rf_importance(X_use, y_tr, seed=seed, n_jobs=n_jobs)
    elif method == "svm_rfe":
        order = rank_svm_rfe(X_use, y_tr, k_min=1, seed=seed)
    elif method == "stability_selection":
        order, extra = rank_stability_selection(X_use, y_tr, k_max=k_max, seed=seed)
        return Selection(method=method, panels=prefix_panels(order, ks), ranking=np.asarray(order, dtype=int),
                         info={"C": {k: extra["C"] for k in ks}, "q_target": extra["q_target"]})
    elif method == "ktsp":
        order, extra = rank_ktsp(X_use, y_tr, k_max=k_max)
        panels, info = {}, {"pairs": extra["pairs"], "exact": {}}
        for k in ks:
            if k % 2 == 0 and k <= len(order):
                panels[k] = np.asarray(order[:k], dtype=int)
                info["exact"][k] = True
            else:                                       # k lẻ hoặc không đủ cặp: không phải panel k-TSP
                panels[k] = None
                info["exact"][k] = False
        return Selection(method=method, panels=panels, ranking=np.asarray(order, dtype=int), info=info)
    elif method == "abess":
        return select_abess(X_use, y_tr, ks, seed=seed)
    elif method in ("l1_lr", "elastic_net"):
        return select_sparse_exact_k(X_use, y_tr, ks, kind=method, seed=seed)
    elif method in ("ga", "bpso"):
        from .wrappers import run_wrapper
        return run_wrapper(method, X_use, y_tr, ks, seed=seed, **(wrapper_kwargs or {}))
    else:
        raise ValueError(f"phương pháp không hỗ trợ: {method}")
    return Selection(method=method, panels=prefix_panels(order, ks), ranking=np.asarray(order, dtype=int))
