"""Bộ phân loại sau chọn gen và chỉ số (KBTN 5.2; QĐ12/M9: thêm log-loss, Brier; M5: chế độ ít mẫu).

* LR-L2 (chính): StandardScaler + LogisticRegression(class_weight = balanced); C ∈ C_GRID chọn bằng
  CV 3 fold phân tầng BÊN TRONG phần train theo AUROC; ngưỡng cho BAcc/MCC chọn để tối đa BAcc trên dự
  đoán out-of-fold của CV nội bộ (không chọn trên test); fit lại trên toàn bộ phần train.
* RF và SVM tuyến tính (phụ, chỉ k ∈ {50, 200}): cùng quy trình chọn ngưỡng.
* B0: cùng LR-L2 trên toàn bộ gen (không lọc sơ bộ).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from . import config as C

CLASSIFIERS = ("lr_l2", "rf", "svm_linear")


@dataclass
class EvalResult:
    classifier: str
    auroc: float
    bacc: float
    mcc: float
    threshold: float
    hyperparam: float | None
    scores_test: np.ndarray
    logloss: float = float("nan")     # chỉ khi điểm là xác suất (LR-L2, RF); SVM tuyến tính: NaN (M9)
    brier: float = float("nan")


def probability_metrics(y: np.ndarray, s: np.ndarray, is_probability: bool) -> tuple[float, float]:
    """log-loss và Brier (M9, QĐ12) khi điểm là xác suất; NaN nếu không (decision_function của SVM)."""
    if not is_probability:
        return float("nan"), float("nan")
    p = np.clip(np.asarray(s, dtype=float), 1e-6, 1 - 1e-6)
    return float(log_loss(y, p, labels=[0, 1])), float(brier_score_loss(y, p))


def _make(classifier: str, c: float | None, seed: int, n_jobs: int):
    if classifier == "lr_l2":
        return ("scale", LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=5000,
                                            random_state=seed))
    if classifier == "svm_linear":
        return ("scale", LinearSVC(C=c, class_weight="balanced", dual="auto", max_iter=10000,
                                   random_state=seed))   # v3.1.6: k < n -> primal (tất định); random_state để an toàn
    if classifier == "rf":
        return (None, RandomForestClassifier(n_estimators=C.RF_N_TREES, class_weight="balanced_subsample",
                                             random_state=seed, n_jobs=n_jobs))
    raise ValueError(classifier)


def _fit_predict(classifier, c, seed, n_jobs, Xtr, ytr, Xte):
    scale, model = _make(classifier, c, seed, n_jobs)
    if scale:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xtr, ytr)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xte)[:, 1]
    return model.decision_function(Xte)


def _best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Ngưỡng tối đa BAcc trên điểm out-of-fold (xét trung điểm giữa các điểm số đã sắp)."""
    order = np.sort(np.unique(s))
    if len(order) == 1:
        return float(order[0])
    cands = (order[:-1] + order[1:]) / 2.0
    best_t, best_b = cands[0], -1.0
    for t in cands:
        b = balanced_accuracy_score(y, (s >= t).astype(int))
        if b > best_b:
            best_t, best_b = float(t), b
    return best_t


def evaluate_panel(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
                   classifier: str = "lr_l2", seed: int = 0, c_grid=C.C_GRID,
                   inner_folds: int = C.INNER_CV_FOLDS, n_jobs: int = 1, small_n: bool | None = None) -> EvalResult:
    """Huấn luyện trên (Xtr, ytr) với CV nội bộ chọn C và ngưỡng; đánh giá trên (Xte, yte).

    small_n (M5): khi mỗi lớp có <= SMALL_N_T_MAX mẫu huấn luyện (mặc định tự phát hiện), C = SMALL_N_FIXED_C cố định,
    không CV nội bộ, ngưỡng = 0,5 trên xác suất (LR-L2, RF) hoặc 0 trên decision_function (SVM).
    """
    is_prob = classifier in ("lr_l2", "rf")
    n_min_class = int(min(np.sum(ytr == 0), np.sum(ytr == 1)))
    if small_n is None:
        small_n = n_min_class <= C.SMALL_N_T_MAX
    if small_n or n_min_class < inner_folds:
        c_fixed = C.SMALL_N_FIXED_C if classifier in ("lr_l2", "svm_linear") else None
        s_te = _fit_predict(classifier, c_fixed, seed, n_jobs, Xtr, ytr, Xte)
        thr = 0.5 if is_prob else 0.0
        pred = (s_te >= thr).astype(int)
        ll, br = probability_metrics(yte, s_te, is_prob)
        return EvalResult(classifier=classifier, auroc=float(roc_auc_score(yte, s_te)),
                          bacc=float(balanced_accuracy_score(yte, pred)), mcc=float(matthews_corrcoef(yte, pred)),
                          threshold=thr, hyperparam=c_fixed, scores_test=s_te, logloss=ll, brier=br)
    inner = list(StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
                 .split(np.zeros(len(ytr)), ytr))
    grid = list(c_grid) if classifier in ("lr_l2", "svm_linear") else [None]
    best_c, best_auc, best_oof = None, -np.inf, None
    for c in grid:
        oof = np.zeros(len(ytr))
        for tr, va in inner:
            oof[va] = _fit_predict(classifier, c, seed, n_jobs, Xtr[tr], ytr[tr], Xtr[va])
        auc = roc_auc_score(ytr, oof)
        if auc > best_auc:
            best_c, best_auc, best_oof = c, auc, oof
    thr = _best_threshold(ytr, best_oof)
    s_te = _fit_predict(classifier, best_c, seed, n_jobs, Xtr, ytr, Xte)
    pred = (s_te >= thr).astype(int)
    ll, br = probability_metrics(yte, s_te, is_prob)
    return EvalResult(classifier=classifier, auroc=float(roc_auc_score(yte, s_te)),
                      bacc=float(balanced_accuracy_score(yte, pred)), mcc=float(matthews_corrcoef(yte, pred)),
                      threshold=thr, hyperparam=best_c, scores_test=s_te, logloss=ll, brier=br)
