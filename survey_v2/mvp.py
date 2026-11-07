"""MVP bài 2 (QĐ3; KBTN v2.2 Mục 6; Review 03/11 Mục 7.3): L1-LR có trọng số phạt từ độ ổn định chọn gen trên nguồn.

π_j = tần suất chọn gen j bằng stability selection L1-LR trên nguồn; w_j = (π_j + 0,05)^(−γ). Phạt Σ w_j·|β_j| tương đương
L1 thường trên X_j / w_j (β'_j = w_j·β_j). Chia cột SAU khi z-score và gọi thẳng select_sparse_exact_k: select() z-score lại
nên sẽ triệt tiêu trọng số.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from . import config as C
from .classify import evaluate_panel, probability_metrics
from .data import variance_prefilter
from .methods import Selection, rank_stability_selection, select_sparse_exact_k, zscore_apply, zscore_fit

PI_EPSILON = 0.05
SOURCE_K_MAX = 10              # q ≈ 2·k_max hệ số của stability selection; cố định, không phụ thuộc --k
SOURCE_SEED_TAG = 8            # seed_for(cid, 0, 0, 8), Review 03/11 Bảng 8 bước 2
GAMMA_MARGIN = 0.01            # M5: γ lớn hơn chỉ thắng khi giảm log-loss OOF ≥ 0,01
INNER_FOLDS = 5
INNER_REPEATS_LARGE_N = 3      # n_T ≥ 20 mỗi lớp: CV 5 fold lặp 3 lần
EARLY_TARGET_FRACTION = 0.30   # thử nghiệm sớm: 70 % nguồn giả / 30 % đích giả theo bệnh nhân (Review 03/11 Bảng 8 bước 8)
EARLY_SEED = 42
BASE_C_GRID = np.logspace(-3, 2, 26)   # lưới mặc định của select_sparse_exact_k (dùng ở E1b)
C_GRID_MAX_LOG_STEP = 0.2              # bước log10 của phần mở rộng, bằng bước của BASE_C_GRID
L1_C_GRID_RULE = "logspace(-3, 2, 26); phần mở rộng tới 100 * max(w_j) với cùng bước khi max(w_j) > 1"


def penalty_weights(pi: np.ndarray, gamma: float, eps: float = PI_EPSILON) -> np.ndarray:
    return (np.asarray(pi, dtype=float) + eps) ** (-float(gamma))


def source_stability(ctx, cid: int) -> tuple[np.ndarray, float]:
    """π_j trên nguồn, theo thứ tự cột của ctx.pre; một lần cho mỗi không gian gen G. Trả về (π, C_q)."""
    Xs = ctx.Xs[:, ctx.pre]
    mu, sd = zscore_fit(Xs)
    _, info = rank_stability_selection(zscore_apply(Xs, mu, sd), ctx.ys, k_max=SOURCE_K_MAX, seed=C.seed_for(cid, 0, 0, SOURCE_SEED_TAG))
    return np.asarray(info["pi"], dtype=float), float(info["C"])


def weighted_c_grid(w: np.ndarray) -> np.ndarray:
    """Lưới C cho L1 trên X / w. Gen có trọng số w_j cần C lớn gấp w_j mới chịu mức phạt như γ = 0 tại C đó, nên trần là 100·max(w_j).

    Nếu giữ trần 100, L1 không với tới các gen có π nhỏ và panel bị F-test bổ sung; F-test không đổi khi chia cột nên bỏ qua π.
    BASE_C_GRID đứng đầu và không đổi: γ = 0 trùng E1b, k đạt được trên lưới cũ thì panel không đổi.
    """
    extra = float(np.log10(np.max(w)))
    if extra <= 0:
        return BASE_C_GRID
    n = int(np.ceil(extra / C_GRID_MAX_LOG_STEP))
    return np.concatenate([BASE_C_GRID, np.logspace(2, 2 + extra, n + 1)[1:]])


def select_weighted(X: np.ndarray, y: np.ndarray, pi: np.ndarray, gamma: float, k_list, seed: int) -> Selection:
    """Chọn đúng k gen trên L_T: z-score theo L_T (M7), chia cột cho w_j, L1-LR đúng k. Chỉ số cột theo ma trận X đưa vào."""
    mu, sd = zscore_fit(X)
    w = penalty_weights(pi, gamma)
    return select_sparse_exact_k(zscore_apply(X, mu, sd) / w, y, k_list, "l1_lr", seed=seed, c_grid=weighted_c_grid(w))


def early_split(ds, target_fraction: float = EARLY_TARGET_FRACTION, seed: int = EARLY_SEED) -> tuple:
    """Chia một bộ thành (nguồn giả, đích giả) theo bệnh nhân, phân tầng theo nhãn. Thứ tự mẫu và gen của bộ gốc được giữ."""
    patients, first = np.unique(ds.groups, return_index=True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=target_fraction, random_state=seed)
    _, target_idx = next(sss.split(np.zeros(len(patients)), ds.y[first]))
    is_target = np.isin(ds.groups, patients[target_idx])

    def part(mask, suffix):
        return replace(ds, name=f"{ds.name}_{suffix}", X=np.ascontiguousarray(ds.X[mask]), y=ds.y[mask], groups=ds.groups[mask],
                       sample_ids=ds.sample_ids[mask], notes={**ds.notes, "early_split": suffix, "early_seed": seed})

    return part(~is_target, "early_source"), part(is_target, "early_target")


def pick_gamma(logloss: dict, margin: float = GAMMA_MARGIN) -> float:
    """M5: xét γ tăng dần, γ lớn hơn thay γ đang giữ chỉ khi giảm log-loss OOF ít nhất `margin`."""
    best = None
    for gamma in sorted(logloss):
        if best is None or logloss[best] - logloss[gamma] >= margin - 1e-12:
            best = gamma
    return best


def pick_gamma_positive(logloss: dict, margin: float = GAMMA_MARGIN) -> float | None:
    """QĐ49: M5 chỉ trên γ > 0 (quy tắc m5_pos). γ = 0 là đối chứng target-only nên không phải ứng viên. Không có γ > 0 thì trả None."""
    return pick_gamma({g: v for g, v in logloss.items() if g > 0}, margin)


def inner_splits(y: np.ndarray, seed: int) -> list:
    """CV phân tầng trong L_T: n_T ≤ 10 mỗi lớp 5 fold một lần, lớn hơn 5 fold lặp 3 lần. Cùng seed cho cùng các fold ở mọi γ."""
    n_min = int(min(np.sum(y == 0), np.sum(y == 1)))
    repeats = 1 if n_min <= C.SMALL_N_T_MAX else INNER_REPEATS_LARGE_N
    splits = []
    for repeat in range(repeats):
        cv = StratifiedKFold(n_splits=min(INNER_FOLDS, n_min), shuffle=True, random_state=C.seed_for(seed, repeat))
        splits += list(cv.split(np.zeros(len(y)), y))
    return splits


def gamma_path(X: np.ndarray, y: np.ndarray, pi: np.ndarray, pre_source: np.ndarray, gammas, k_list, prefilter: int, seed: int) -> dict:
    """Chỉ số OOF cho mọi (γ, k), chỉ dùng L_T. Trong mỗi fold trong: lọc (γ = 0), chọn gen lại, fit LR-L2 với C = 1.

    X là L_T trên toàn G; `pi` theo thứ tự `pre_source`. γ = 0 lọc phương sai trên phần train của fold như E1b trên L_T, γ > 0 giữ
    bộ lọc trên nguồn. Trả về {(γ, k): {oof_logloss, oof_brier, oof_auroc, n_folds, n_repeats}}.
    """
    splits = inner_splits(y, seed)
    n_folds = min(INNER_FOLDS, int(min(np.sum(y == 0), np.sum(y == 1))))
    scores = {}
    for f, (tr, va) in enumerate(splits):
        fold_seed = C.seed_for(seed, 1000 + f)                    # chung cho mọi γ để ghép cặp
        for gamma in gammas:
            if gamma == 0:
                pre, pi_pre = variance_prefilter(X[tr], prefilter), None
            else:
                pre, pi_pre = pre_source, pi
            ks = [k for k in k_list if k <= len(pre)]
            sel = select_weighted(X[tr][:, pre], y[tr], np.zeros(len(pre)) if pi_pre is None else pi_pre, gamma, ks, fold_seed)
            for k in ks:
                cols = pre[np.asarray(sel.panels[k])]
                res = evaluate_panel(X[tr][:, cols], y[tr], X[va][:, cols], y[va], classifier="lr_l2", seed=fold_seed, small_n=True)
                scores.setdefault((gamma, k), []).append(res.scores_test)
    y_oof = np.concatenate([y[va] for _, va in splits])
    path = {}
    for key, parts in scores.items():
        s = np.concatenate(parts)
        logloss, brier = probability_metrics(y_oof, s, True)
        path[key] = {"oof_logloss": logloss, "oof_brier": brier, "oof_auroc": float(roc_auc_score(y_oof, s)),
                     "n_folds": n_folds, "n_repeats": len(splits) // n_folds}
    return path
