"""Cổng bằng nhau của RQ2 (QĐ51; Review 07/11 Bảng 7 bước 5): dòng đối chứng phải trùng E1b, E7 trước khi tính thống kê."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
COHORT = "gse37745"


def _gate():
    spec = importlib.util.spec_from_file_location("check_rq2_equality", REPO / "scripts" / "check_rq2_equality.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _metrics(base: float) -> dict:
    return {"auroc": base, "bacc": base - 0.1, "mcc": base - 0.2, "logloss": base + 0.1, "brier": base / 10}


def _write(root: Path, n_t=("5", "10")):
    """Bộ tệp nhỏ viết tay, hai mức n_T, k = 2 và 5; mỗi nhánh một giá trị riêng để ghép nhầm nhánh là lộ ngay."""
    key = lambda nt, k: {"cohort": COHORT, "split": 0, "seed": 42, "subset": 0, "n_t": nt, "k": k}
    rq2, e1b, e7 = [], [], []
    for nt in n_t:
        for k in (2, 5):
            target = _metrics(0.91 + k / 100)
            rq2.append({**key(nt, k), "method": "mvp", "gamma": 0.0, "gamma_rule": "fixed", "prefilter_space": "target", **target})
            rq2.append({**key(nt, k), "method": "mvp", "gamma": 0.0, "gamma_rule": "fixed", "prefilter_space": "source", **_metrics(0.5)})
            rq2.append({**key(nt, k), "method": "mvp", "gamma": 1.0, "gamma_rule": "fixed", "prefilter_space": "source", **_metrics(0.6)})
            e1b.append({**key(nt, k), "method": "l1_lr", **target})
            e1b.append({**key(nt, k), "method": "f_test", **_metrics(0.1)})
            for strategy, base in (("select_on_source", 0.81), ("pooled", 0.71)):
                m = _metrics(base + k / 100)
                rq2.append({**key(nt, k), "method": strategy, "gamma": float("nan"), "gamma_rule": "none", "prefilter_space": "source", **m})
                e7.append({**key(nt, k), "strategy": strategy, **m})
            e7.append({**key(nt, k), "strategy": "target_only", **_metrics(0.2)})
        for method, k, base in (("ihc_2", 2, 0.61), ("ihc_4", 4, 0.62)):
            m = _metrics(base)
            rq2.append({**key(nt, k), "method": method, "gamma": float("nan"), "gamma_rule": "none", "prefilter_space": "none", **m})
            e1b.append({**key(nt, k), "method": method, **m})
    paths = {"rq2": root / "RQ2_mvp" / COHORT / "rq2_mvp_per_run.csv", "e1b": root / "E1b" / COHORT / "e1b_per_run.csv", "e7": root / "E7" / COHORT / "e7_per_run.csv"}
    for name, rows in (("rq2", rq2), ("e1b", e1b), ("e7", e7)):
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(paths[name], index=False)
    return paths


def _run(root):
    return _gate().main(["--results-root", str(root), "--datasets", COHORT])


def _edit(path, query, col, delta):
    df = pd.read_csv(path, dtype={"n_t": str})
    idx = df.query(query).index
    assert len(idx) == 1
    df.loc[idx, col] += delta
    df.to_csv(path, index=False)


def test_identical_files_pass_and_report_every_pair(tmp_path):
    _write(tmp_path)
    assert _run(tmp_path) == 0
    report = pd.read_csv(tmp_path / "RQ2_mvp" / "rq2_equality_report.csv", dtype={"n_t": str})
    assert set(report.status) == {"ok"}
    # 2 mức n_T × (target 2 k + D3 2 chiến lược × 2 k + IHC 2 panel) = 2 × 8
    assert report.groupby("arm").size().to_dict() == {"ihc_2": 2, "ihc_4": 2, "pooled": 4, "select_on_source": 4, "target_only": 4}


@pytest.mark.parametrize("which, query, col", [
    ("rq2", "method == 'mvp' and prefilter_space == 'target' and n_t == '5' and k == 2", "auroc"),
    ("rq2", "method == 'pooled' and n_t == '10' and k == 5", "brier"),
    ("rq2", "method == 'ihc_4' and n_t == '5'", "mcc"),
    ("e7", "strategy == 'select_on_source' and n_t == '5' and k == 2", "logloss"),
    ("e1b", "method == 'l1_lr' and n_t == '10' and k == 5", "bacc"),
])
def test_difference_of_1e_9_is_blocked(tmp_path, which, query, col):
    """Review 07/11: một dòng bị sửa 10⁻⁹ thì cổng phải chặn, ở bất kỳ nhánh, chỉ số hay phía nào."""
    paths = _write(tmp_path)
    _edit(paths[which], query, col, 1e-9)
    assert _run(tmp_path) != 0
    report = pd.read_csv(tmp_path / "RQ2_mvp" / "rq2_equality_report.csv", dtype={"n_t": str})
    assert (report.status == "diff").sum() == 1 and report.loc[report.status == "diff", "worst_metric"].iloc[0] == col


def test_difference_below_tolerance_passes(tmp_path):
    paths = _write(tmp_path)
    _edit(paths["rq2"], "method == 'pooled' and n_t == '10' and k == 5", "auroc", 5e-13)
    assert _run(tmp_path) == 0


def test_mechanism_arm_and_other_methods_are_not_compared(tmp_path):
    """Nhánh γ = 0/source và γ > 0 không có đối chứng, sửa chúng không được làm cổng đổi kết quả."""
    paths = _write(tmp_path)
    _edit(paths["rq2"], "method == 'mvp' and prefilter_space == 'source' and gamma == 0 and n_t == '5' and k == 2", "auroc", 0.1)
    _edit(paths["rq2"], "method == 'mvp' and gamma == 1 and n_t == '5' and k == 2", "auroc", 0.1)
    assert _run(tmp_path) == 0


def test_row_missing_from_rq2_is_blocked(tmp_path):
    """Thiếu dòng so với đối chứng không được coi là đạt, nếu không một phiên chạy dở sẽ qua cổng."""
    paths = _write(tmp_path)
    df = pd.read_csv(paths["rq2"], dtype={"n_t": str})
    df[~((df.method == "pooled") & (df.n_t == "10") & (df.k == 5))].to_csv(paths["rq2"], index=False)
    assert _run(tmp_path) != 0
    report = pd.read_csv(tmp_path / "RQ2_mvp" / "rq2_equality_report.csv", dtype={"n_t": str})
    assert report.status.value_counts().to_dict().get("missing_in_rq2") == 1


def test_row_missing_from_reference_is_blocked(tmp_path):
    paths = _write(tmp_path)
    df = pd.read_csv(paths["e7"], dtype={"n_t": str})
    df[~((df.strategy == "pooled") & (df.n_t == "10") & (df.k == 5))].to_csv(paths["e7"], index=False)
    assert _run(tmp_path) != 0
    report = pd.read_csv(tmp_path / "RQ2_mvp" / "rq2_equality_report.csv", dtype={"n_t": str})
    assert report.status.value_counts().to_dict().get("missing_in_reference") == 1


def test_duplicate_row_is_blocked(tmp_path):
    paths = _write(tmp_path)
    df = pd.read_csv(paths["rq2"], dtype={"n_t": str})
    pd.concat([df, df[(df.method == "ihc_2") & (df.n_t == "5")]]).to_csv(paths["rq2"], index=False)
    assert _run(tmp_path) != 0


def test_missing_file_is_blocked(tmp_path):
    paths = _write(tmp_path)
    paths["e7"].unlink()
    assert _run(tmp_path) != 0


def test_nan_on_one_side_is_blocked_and_on_both_sides_passes(tmp_path):
    paths = _write(tmp_path)
    for name in ("rq2", "e7"):
        df = pd.read_csv(paths[name], dtype={"n_t": str})
        col = "method" if name == "rq2" else "strategy"
        df.loc[(df[col] == "pooled") & (df.n_t == "5") & (df.k == 2), "mcc"] = float("nan")
        df.to_csv(paths[name], index=False)
    assert _run(tmp_path) == 0
    df = pd.read_csv(paths["e7"], dtype={"n_t": str})
    df.loc[(df.strategy == "pooled") & (df.n_t == "5") & (df.k == 2), "mcc"] = 0.5
    df.to_csv(paths["e7"], index=False)
    assert _run(tmp_path) != 0


def test_n_t_option_limits_the_scope(tmp_path):
    """Đối chứng có mức n_T = 10 mà RQ2 chỉ chạy n_T = 5: mặc định chặn, chỉ qua khi khai báo rõ phạm vi."""
    paths = _write(tmp_path)
    df = pd.read_csv(paths["rq2"], dtype={"n_t": str})
    df[df.n_t == "5"].to_csv(paths["rq2"], index=False)
    assert _run(tmp_path) != 0
    assert _gate().main(["--results-root", str(tmp_path), "--datasets", COHORT, "--n-t", "5"]) == 0


@pytest.mark.slow
def test_real_scripts_pass_the_gate_end_to_end(tmp_path_factory, monkeypatch):
    """Chạy thật E1b, E7 và RQ2 trên dữ liệu giả rồi qua cổng. Hỏng nếu seed hoặc hàm của một nhánh lệch E1b, E7."""
    import numpy as np

    root = tmp_path_factory.mktemp("kltn_gate")
    monkeypatch.setenv("KLTN_DATA_ROOT", str(root))
    monkeypatch.setenv("KLTN_ALLOW_UNPUSHED", "1")
    monkeypatch.setenv("KLTN_ALLOW_UNTRACKED", "1")
    monkeypatch.setenv("KLTN_STRICT_N", "0")
    for m in list(sys.modules):
        if m.startswith("scripts.") or m == "scripts" or m.startswith("survey_v2.runlog") or m.startswith("run_") or m == "provenance":
            del sys.modules[m]
    sys.path.insert(0, str(REPO / "scripts"))
    from survey_v2 import config as C, synthetic
    from survey_v2.data import load_dataset
    from survey_v2.fewshot import build_fewshot_manifest

    rng = np.random.default_rng(2026)
    informative = rng.choice(400, size=15, replace=False)
    genes = np.array([f"G{j:05d}" for j in range(400)])
    genes[informative[:4]] = ["NKX2-1", "TP63", "NAPSA", "KRT5"]
    synthetic.make_dataset(root, C.SOURCE_DATASET, n=160, p=400, effect=0.9, seed=10, informative=informative, genes=genes, with_folds=False, prevalence=0.5)
    synthetic.make_geo_target(root, COHORT, genes, genes[informative], n=90, effect=0.9, seed=22)
    ds = load_dataset(root, COHORT)
    build_fewshot_manifest(COHORT, ds.sample_ids, ds.groups, ds.y).to_csv(ds.data_dir / "fewshot_splits.csv", index=False)

    import importlib

    common = ["--datasets", COHORT, "--k", "2", "5", "--n-t", "5", "--prefilter", "120"]
    importlib.import_module("run_survey_e1b_learning_curve").main([*common, "--methods", "l1_lr", "ihc_2", "ihc_4", "--random-draws", "1"])
    importlib.import_module("run_survey_e7_transfer_naive").main([*common, "--strategies", "select_on_source", "pooled"])
    importlib.import_module("run_rq2_mvp").main([*common, "--gamma", "0", "1", "--out", "RQ2_mvp"])
    assert _gate().main(["--results-root", str(root / "results" / "survey_v2"), "--datasets", COHORT, "--n-t", "5"]) == 0
