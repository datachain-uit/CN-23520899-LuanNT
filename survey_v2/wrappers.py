"""Wrapper GA và BPSO với mã hóa ĐÚNG k gen; fitness = AUROC CV 3 fold của LR-L2 bên trong phần train.

Sửa O4 (Review 04/10): fitness không dùng một holdout cố định mà dùng CV nội bộ lặp lại trên
phần train của fold; phần test không bao giờ được đọc trong quá trình tìm kiếm. Ngân sách tìm kiếm
cố định (quần thể P = 30, T = 50 thế hệ, Review 24/10 M10); mọi lần đánh giá fitness được đếm
(`n_eval`) để báo cáo chi phí không phụ thuộc phần cứng (O7).

GA (Siedlecki & Sklansky, 1989; mã hóa tập chỉ số kích thước k):
    chọn lọc tournament (3) → lai ghép giữ giao của cha mẹ rồi bổ sung ngẫu nhiên từ hiệu đối xứng →
    đột biến đổi gen với xác suất 1/k mỗi gen → giữ 2 cá thể tốt nhất (elitism).
BPSO (Kennedy & Eberhart, 1997) với sửa chữa đúng k:
    vận tốc v cập nhật theo pbest/gbest; x_j = 1 với xác suất sigmoid(v_j); sau đó sửa về đúng k gen
    (thừa: bỏ gen có sigmoid nhỏ nhất; thiếu: thêm gen có sigmoid lớn nhất).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import config as C
from .methods import Selection


class InnerCVFitness:
    """AUROC trung bình của LR-L2 (C cố định) trên CV k-fold phân tầng BÊN TRONG phần train."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seed: int, n_folds: int = C.WRAPPER_FITNESS_CV,
                 C_lr: float = C.WRAPPER_FITNESS_C):
        self.X, self.y = X, y
        self.C_lr = C_lr
        self.seed = int(seed)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        self.folds = list(cv.split(np.zeros(len(y)), y))
        self.n_eval = 0
        self._cache: dict[tuple, float] = {}

    def __call__(self, genes: np.ndarray) -> float:
        key = tuple(sorted(int(g) for g in genes))
        if key in self._cache:
            return self._cache[key]
        self.n_eval += 1
        Xs = self.X[:, list(key)]
        scores = []
        for tr, va in self.folds:
            sc = StandardScaler().fit(Xs[tr])
            m = LogisticRegression(C=self.C_lr, class_weight="balanced", solver="liblinear", max_iter=5000,
                                   random_state=self.seed)   # v3.1.6: primal L2-LR tất định; random_state để an toàn
            m.fit(sc.transform(Xs[tr]), self.y[tr])
            scores.append(roc_auc_score(self.y[va], m.decision_function(sc.transform(Xs[va]))))
        val = float(np.mean(scores))
        self._cache[key] = val
        return val


# ---------------------------------------------------------------------------------------------------
# GA
# ---------------------------------------------------------------------------------------------------
def _ga_crossover(a: np.ndarray, b: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    common = np.intersect1d(a, b)
    pool = np.setdiff1d(np.union1d(a, b), common)
    need = k - len(common)
    child = common if need == 0 else np.concatenate([common, rng.choice(pool, size=need, replace=False)])
    return np.sort(child)


def _ga_mutate(ind: np.ndarray, p: int, rng: np.random.Generator, rate: float) -> np.ndarray:
    ind = ind.copy()
    flip = rng.random(len(ind)) < rate
    if flip.any():
        outside = np.setdiff1d(np.arange(p), ind)
        new = rng.choice(outside, size=int(flip.sum()), replace=False)
        ind[flip] = new
    return np.sort(ind)


def genetic_algorithm(fitness, p: int, k: int, rng: np.random.Generator, pop: int = C.WRAPPER_POP,
                      generations: int = C.WRAPPER_GENERATIONS, tournament: int = C.GA_TOURNAMENT,
                      elite: int = C.GA_ELITE, mutation_per_gene: float | None = C.GA_MUTATION_PER_GENE):
    rate = (1.0 / k) if mutation_per_gene is None else mutation_per_gene
    population = [np.sort(rng.choice(p, size=k, replace=False)) for _ in range(pop)]
    fit = np.array([fitness(ind) for ind in population])
    history = [float(fit.max())]
    for _ in range(generations):
        order = np.argsort(-fit, kind="stable")
        new_pop = [population[i] for i in order[:elite]]
        while len(new_pop) < pop:
            cand = rng.choice(pop, size=tournament, replace=False)
            pa = population[cand[np.argmax(fit[cand])]]
            cand = rng.choice(pop, size=tournament, replace=False)
            pb = population[cand[np.argmax(fit[cand])]]
            child = _ga_mutate(_ga_crossover(pa, pb, k, rng), p, rng, rate)
            new_pop.append(child)
        population = new_pop
        fit = np.array([fitness(ind) for ind in population])
        history.append(float(fit.max()))
    best = population[int(np.argmax(fit))]
    return np.sort(best), float(fit.max()), history


# ---------------------------------------------------------------------------------------------------
# BPSO
# ---------------------------------------------------------------------------------------------------
def _sigmoid(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-v))


def _repair_to_k(x: np.ndarray, s: np.ndarray, k: int) -> np.ndarray:
    """Sửa vector nhị phân x về đúng k số 1 dựa trên xác suất s."""
    ones = np.flatnonzero(x)
    if len(ones) > k:
        drop = ones[np.argsort(s[ones], kind="stable")[: len(ones) - k]]
        x[drop] = 0
    elif len(ones) < k:
        zeros = np.flatnonzero(x == 0)
        add = zeros[np.argsort(-s[zeros], kind="stable")[: k - len(ones)]]
        x[add] = 1
    return x


def binary_pso(fitness, p: int, k: int, rng: np.random.Generator, pop: int = C.WRAPPER_POP,
               iterations: int = C.WRAPPER_GENERATIONS, w_start: float = C.BPSO_W_START,
               w_end: float = C.BPSO_W_END, c1: float = C.BPSO_C1, c2: float = C.BPSO_C2,
               v_max: float = C.BPSO_V_MAX):
    X = np.zeros((pop, p), dtype=np.int8)
    for i in range(pop):
        X[i, rng.choice(p, size=k, replace=False)] = 1
    V = rng.uniform(-1, 1, size=(pop, p))
    fit = np.array([fitness(np.flatnonzero(X[i])) for i in range(pop)])
    pbest, pbest_fit = X.copy(), fit.copy()
    g = int(np.argmax(fit))
    gbest, gbest_fit = X[g].copy(), float(fit[g])
    history = [gbest_fit]
    for t in range(iterations):
        w = w_start - (w_start - w_end) * t / max(iterations - 1, 1)
        r1, r2 = rng.random((pop, p)), rng.random((pop, p))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest[None, :] - X)
        V = np.clip(V, -v_max, v_max)
        S = _sigmoid(V)
        X = (rng.random((pop, p)) < S).astype(np.int8)
        for i in range(pop):
            X[i] = _repair_to_k(X[i], S[i], k)
        fit = np.array([fitness(np.flatnonzero(X[i])) for i in range(pop)])
        improved = fit > pbest_fit
        pbest[improved], pbest_fit[improved] = X[improved], fit[improved]
        g = int(np.argmax(pbest_fit))
        if pbest_fit[g] > gbest_fit:
            gbest, gbest_fit = pbest[g].copy(), float(pbest_fit[g])
        history.append(gbest_fit)
    return np.flatnonzero(gbest), gbest_fit, history


# ---------------------------------------------------------------------------------------------------
# Giao diện
# ---------------------------------------------------------------------------------------------------
def run_wrapper(method: str, X_tr: np.ndarray, y_tr: np.ndarray, k_list, seed: int = 0,
                pop: int = C.WRAPPER_POP, generations: int = C.WRAPPER_GENERATIONS) -> Selection:
    """Chạy GA/BPSO cho từng k trên phần train; trả về panels, n_eval và lịch sử fitness."""
    p = X_tr.shape[1]
    panels, info = {}, {"n_eval": {}, "best_fitness": {}, "history": {}, "pop": pop, "generations": generations}
    for k in sorted(int(k) for k in k_list):
        rng = np.random.default_rng([seed, k, 0 if method == "ga" else 1])
        fitness = InnerCVFitness(X_tr, y_tr, seed=seed)
        if method == "ga":
            best, best_fit, hist = genetic_algorithm(fitness, p, k, rng, pop=pop, generations=generations)
        elif method == "bpso":
            best, best_fit, hist = binary_pso(fitness, p, k, rng, pop=pop, iterations=generations)
        else:
            raise ValueError(method)
        assert len(best) == k and len(set(best.tolist())) == k
        panels[k] = np.asarray(best, dtype=int)
        info["n_eval"][k], info["best_fitness"][k], info["history"][k] = fitness.n_eval, best_fit, hist
    return Selection(method=method, panels=panels, info=info)
