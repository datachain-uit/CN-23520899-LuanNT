"""scripts/w5_bas_vs_sqc.py (W5; Review 09/11 V4, Y81, QĐ96): BAS gần tâm SQC hay tâm ADC, gen chọn không dùng nhãn."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load():
    spec = importlib.util.spec_from_file_location("w5_under_test", REPO / "scripts" / "w5_bas_vs_sqc.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load()


def _data(seed=0, n_adc=20, n_sqc=15, n_bas=6, p=60, bas_like="SQC"):
    """10 gen tách lớp mạnh (ADC thấp, SQC cao); BAS lấy hồ sơ của bas_like; còn lại nhiễu."""
    rng = np.random.default_rng(seed)
    hist = np.array(["ADC"] * n_adc + ["SQC"] * n_sqc + ["BAS"] * n_bas)
    X = rng.normal(8, 0.3, (len(hist), p))
    shift = np.where(hist == "ADC", -2.0, 2.0)
    if bas_like == "ADC":
        shift[hist == "BAS"] = -2.0
    X[:, :10] += shift[:, None]
    return X, hist, [f"s{i}" for i in range(len(hist))]


def test_top_variance_genes_ignore_labels_and_pick_largest():
    X = np.column_stack([np.zeros(5), np.arange(5.0), 3 * np.arange(5.0), np.ones(5)])
    assert m.top_variance_genes(X, 2).tolist() == [1, 2]
    with pytest.raises(ValueError):
        m.top_variance_genes(X, 5)


def test_bas_with_sqc_profile_is_closer_to_sqc_and_adc_profile_to_adc():
    X, h, ids = _data(bas_like="SQC")
    per = m.per_sample(X, h, ids, 20)
    bas = per[per.histology == "BAS"]
    assert (bas.d_r > 0).all() and (bas.d_e > 0).all() and (bas.closer_to == "SQC").all()
    X, h, ids = _data(bas_like="ADC")
    bas = m.per_sample(X, h, ids, 20).query("histology == 'BAS'")
    assert (bas.d_r < 0).all() and (bas.d_e < 0).all()


def test_bas_samples_do_not_enter_centroids():
    """Đổi giá trị mẫu BAS (cả phần trừ trung bình được giữ cố định) không đổi tâm: r của mẫu ADC/SQC chỉ đổi qua bước trừ trung bình."""
    X, h, ids = _data()
    Z = X - X.mean(axis=0)
    per = m.per_sample(X, h, ids, X.shape[1])
    j = int(np.flatnonzero(h == "SQC")[0])
    sqc_rest = [i for i in np.flatnonzero(h == "SQC") if i != j]
    adc = np.flatnonzero(h == "ADC")
    r_sqc = np.corrcoef(Z[j], Z[sqc_rest].mean(axis=0))[0, 1]
    r_adc = np.corrcoef(Z[j], Z[adc].mean(axis=0))[0, 1]
    row = per.iloc[j]
    assert row.r_sqc == pytest.approx(r_sqc, abs=1e-12)
    assert row.r_adc == pytest.approx(r_adc, abs=1e-12)


def test_leave_one_out_centroid_for_own_group():
    X, h, ids = _data()
    Z = X - X.mean(axis=0)
    per = m.per_sample(X, h, ids, X.shape[1])
    j = int(np.flatnonzero(h == "ADC")[3])
    rest = [i for i in np.flatnonzero(h == "ADC") if i != j]
    expect = np.linalg.norm(Z[j] - Z[rest].mean(axis=0))
    assert per.iloc[j].dist_adc == pytest.approx(expect, abs=1e-10)


def test_summary_reports_distribution_and_counts():
    X, h, ids = _data()
    per = pd.concat([m.per_sample(X, h, ids, n) for n in (10, 20)], ignore_index=True)
    s = m.summarize(per)
    assert set(s.columns) >= {"n", "n_positive", "frac_positive", "min", "q25", "median", "q75", "max", "mean"}
    row = s[(s.n_genes == 20) & (s.metric == "d_r") & (s.histology == "BAS")].iloc[0]
    v = per[(per.n_genes == 20) & (per.histology == "BAS")].d_r
    assert row.n == 6 and row.n_positive == int((v > 0).sum())
    assert row["median"] == pytest.approx(float(v.median()))
    assert s.histology.tolist()[:3] == ["ADC", "SQC", "BAS"]


def test_unknown_histology_stops():
    X, h, ids = _data()
    h = h.copy()
    h[0] = "LCC"
    with pytest.raises(ValueError):
        m.per_sample(X, h, ids, 10)


def test_read_histology_strips_prefix_and_follows_sample_order(tmp_path):
    p = tmp_path / "metadata.csv"
    pd.DataFrame({"sample_id": ["a", "b", "c"], "histology_raw": ["histology: ADC", "histology: BAS", "histology: SQC"]}).to_csv(p, index=False)
    assert m.read_histology(p, ["c", "a", "b"]).tolist() == ["SQC", "ADC", "BAS"]
