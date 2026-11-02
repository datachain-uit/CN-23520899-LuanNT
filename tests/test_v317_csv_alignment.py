"""v3.1.7 (Review 02/11/2026, Y31): nối CSV theo tiêu đề đã có; sửa dòng cũ lệch cột (n_features_vs_k nằm ở cột n_draws)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_e1_classification import append_csv_aligned, normalise_per_fold  # noqa: E402


def test_append_csv_aligned_keeps_header_order_and_extends(tmp_path):
    p = tmp_path / "x.csv"
    append_csv_aligned(pd.DataFrame([{"a": 1, "b": 2, "n_draws": 7}]), p)
    append_csv_aligned(pd.DataFrame([{"a": 3, "b": 4, "n_features_vs_k": -3}]), p)     # cột mới -> ghi lại cả tệp
    append_csv_aligned(pd.DataFrame([{"b": 6, "a": 5, "n_features_vs_k": 0}]), p)      # thiếu n_draws, thứ tự khác -> theo tiêu đề
    df = pd.read_csv(p)
    assert df.columns.tolist() == ["a", "b", "n_draws", "n_features_vs_k"]
    assert df.a.tolist() == [1, 3, 5] and df.b.tolist() == [2, 4, 6]
    assert df.n_draws.tolist()[0] == 7 and np.isnan(df.n_draws.iloc[1]) and np.isnan(df.n_draws.iloc[2])
    assert np.isnan(df.n_features_vs_k.iloc[0]) and df.n_features_vs_k.tolist()[1:] == [-3, 0]


def test_legacy_misaligned_rows_are_repaired():
    """Trước v3.1.7, dòng panel nối sau dòng random-k bị ghi theo vị trí: giá trị n_features_vs_k rơi vào cột n_draws."""
    header = ["cohort", "repeat", "seed", "fold", "method", "k", "model", "n_features", "auroc", "fit_time_s", "n_draws", "n_features_vs_k"]
    rows = [["c", 0, 42, 0, "random_k", 10, "lr_l2", 10, 0.8, 0.1, 100, np.nan],
            ["c", 0, 42, 0, "b0", 400, "lr_l2", 400, 0.95, 0.5, np.nan, np.nan],
            ["c", 0, 42, 0, "f_test", 10, "lr_l2", 10, 0.9, 0.1, np.nan, 0],          # đúng
            ["c", 0, 42, 0, "svm_rfe", 10, "lr_l2", 10, 0.91, 0.1, 0, np.nan],        # lệch (giá trị 0 ở n_draws)
            ["c", 0, 42, 0, "panel_girard2016", 42, "lr_l2", 39, 0.88, 0.1, -3, np.nan]]  # lệch (−3 ở n_draws)
    df, n = normalise_per_fold(pd.DataFrame(rows, columns=header))
    assert n == 2
    byname = df.set_index("method")
    assert byname.loc["random_k", "n_draws"] == 100 and np.isnan(byname.loc["random_k", "n_features_vs_k"])
    assert np.isnan(byname.loc["svm_rfe", "n_draws"]) and byname.loc["svm_rfe", "n_features_vs_k"] == 0
    assert byname.loc["panel_girard2016", "n_features_vs_k"] == -3 and np.isnan(byname.loc["panel_girard2016", "n_draws"])
    assert byname.loc["f_test", "n_features_vs_k"] == 0 and np.isnan(byname.loc["b0", "n_features_vs_k"])
    assert byname.auroc.tolist() == [0.8, 0.95, 0.9, 0.91, 0.88]                       # chỉ số không đổi
    df2, n2 = normalise_per_fold(df)
    assert n2 == 0 and df2.equals(df)
