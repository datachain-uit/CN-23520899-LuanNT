"""MVP bài 2 (QĐ3; KBTN v2.2 Mục 6; Review 03/11 Mục 7.3): L1-LR có trọng số phạt từ độ ổn định chọn gen trên nguồn.

π_j = tần suất chọn gen j bằng stability selection L1-LR trên nguồn; w_j = (π_j + 0,05)^(−γ). Phạt Σ w_j·|β_j| tương đương
L1 thường trên X_j / w_j (β'_j = w_j·β_j). Chia cột SAU khi z-score và gọi thẳng select_sparse_exact_k: select() z-score lại
nên sẽ triệt tiêu trọng số.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from .methods import Selection, rank_stability_selection, select_sparse_exact_k, zscore_apply, zscore_fit

PI_EPSILON = 0.05
SOURCE_K_MAX = 10              # q ≈ 2·k_max hệ số của stability selection; cố định, không phụ thuộc --k
SOURCE_SEED_TAG = 8            # seed_for(cid, 0, 0, 8), Review 03/11 Bảng 8 bước 2


def penalty_weights(pi: np.ndarray, gamma: float, eps: float = PI_EPSILON) -> np.ndarray:
    return (np.asarray(pi, dtype=float) + eps) ** (-float(gamma))


def source_stability(ctx, cid: int) -> tuple[np.ndarray, float]:
    """π_j trên nguồn, theo thứ tự cột của ctx.pre; một lần cho mỗi không gian gen G. Trả về (π, C_q)."""
    Xs = ctx.Xs[:, ctx.pre]
    mu, sd = zscore_fit(Xs)
    _, info = rank_stability_selection(zscore_apply(Xs, mu, sd), ctx.ys, k_max=SOURCE_K_MAX, seed=C.seed_for(cid, 0, 0, SOURCE_SEED_TAG))
    return np.asarray(info["pi"], dtype=float), float(info["C"])


def select_weighted(X: np.ndarray, y: np.ndarray, pi: np.ndarray, gamma: float, k_list, seed: int) -> Selection:
    """Chọn đúng k gen trên L_T: z-score theo L_T (M7), chia cột cho w_j, L1-LR đúng k. Chỉ số cột theo ma trận X đưa vào."""
    mu, sd = zscore_fit(X)
    return select_sparse_exact_k(zscore_apply(X, mu, sd) / penalty_weights(pi, gamma), y, k_list, "l1_lr", seed=seed)
