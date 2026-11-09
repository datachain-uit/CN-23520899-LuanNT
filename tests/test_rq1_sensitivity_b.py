"""Độ nhạy RQ1 phần (b) (QĐ53b, Review 08/11 Mục 8.5): lọc F-test trong fold, đăng ký biến thể gse30219_bas, việc Kaggle rq1_sens_b."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import ftest_prefilter, prefilter, variance_prefilter  # noqa: E402


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem + "_rq1b", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_ftest_prefilter_keeps_label_associated_genes_and_ignores_variance():
    rng = np.random.default_rng(0)
    y = np.array([0] * 20 + [1] * 20)
    X = rng.normal(size=(40, 50))
    X[:, :5] += 3 * y[:, None]                 # 5 gen tín hiệu, phương sai vừa
    X[:, 40:] *= 10                            # 10 gen nhiễu phương sai rất lớn
    keep = ftest_prefilter(X, y, 5)
    assert list(keep) == [0, 1, 2, 3, 4]
    assert set(variance_prefilter(X, 5)) <= set(range(40, 50))          # lọc phương sai lấy gen nhiễu
    assert list(prefilter(X, y, 5, "ftest")) == list(keep) and list(prefilter(X, y, 5, "variance")) == list(variance_prefilter(X, 5))
    with pytest.raises(ValueError):
        prefilter(X, y, 5, "mi")


def test_ftest_prefilter_is_sorted_stable_and_handles_constant_genes():
    y = np.array([0, 0, 1, 1, 0, 1])
    X = np.zeros((6, 4)); X[:, 2] = y; X[:, 3] = y           # gen 0, 1 hằng (F = NaN), gen 2, 3 hoà
    assert list(ftest_prefilter(X, y, 1)) == [2] and list(ftest_prefilter(X, y, 3)) == [0, 2, 3]
    assert list(ftest_prefilter(X, y, 10)) == [0, 1, 2, 3]


def test_bas_variant_is_registered_outside_the_survey_datasets():
    assert "gse30219_bas" not in C.DATASETS and "gse30219_bas" not in C.DATASET_DIRS
    assert C.dataset_dir("gse30219_bas") == f"{C.GEO_TARGET_DIR}/gse30219_bas"
    assert C.dataset_seed_id("gse30219_bas") == 200 and C.dataset_seed_id("gse30219") == 102           # seed của bộ chính không đổi
    assert C.DATASET_EXPECTED_N["gse30219_bas"] == (81, 99) and C.DATASET_PLATFORM["gse30219_bas"] == "GPL570"


def test_feature_selection_accepts_prefilter_rule_and_defaults_to_variance():
    fs = _load(REPO / "scripts" / "run_e1_feature_selection.py")
    assert fs.parse_args(["--datasets", "gse30219"]).prefilter_rule == "variance"
    args = fs.parse_args(["--datasets", "gse30219", "--prefilter", "2000", "--prefilter-rule", "ftest"])
    assert (args.prefilter, args.prefilter_rule) == (2000, "ftest")


def test_rq1_sens_b_job_runs_both_configurations_without_touching_survey_outputs():
    kj = _load(REPO / "scripts" / "kaggle_jobs.py")
    job = kj.JOBS["rq1_sens_b"]
    fs = _load(REPO / "scripts" / "run_e1_feature_selection.py")
    clf = _load(REPO / "scripts" / "run_e1_classification.py")
    shards = [fs.parse_args(s[1:]) for s in job.shards]
    bas = [a for a in shards if a.datasets == ["gse30219_bas"]]
    ftest = [a for a in shards if a.prefilter_rule == "ftest"]
    assert len(bas) == 1 and bas[0].methods == ["l1_lr", "ihc_4"] and bas[0].out == "E1_feature_selection_bas" and bas[0].prefilter_rule == "variance"
    assert sorted(a.datasets[0] for a in ftest) == sorted(C.GPL570_COHORTS) and all(a.prefilter == 2000 and a.methods == ["l1_lr"] for a in ftest)
    fin = [clf.parse_args(s[1:]) for s in job.finalize]
    assert {a.out for a in fin} == {"E1_classification_bas", "E1_classification_ftest2000"}
    ft = [a for a in fin if a.out == "E1_classification_ftest2000"]
    assert all(not a.references for a in ft) and [a for a in fin if a.out == "E1_classification_bas"][0].references
    outs = set(job.outputs)
    assert not outs & {f"E1_classification/{d}" for d in C.DATASETS} and not outs & {f"E1_feature_selection/{d}" for d in C.DATASETS}
    assert job.restore_from == [] and job.requires == []
