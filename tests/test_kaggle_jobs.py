"""scripts/kaggle_jobs.py: mọi lệnh trong danh mục trỏ tới script có thật và được parser của script chấp nhận; gộp E2/E3."""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem + "_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclass cần module trong sys.modules
    spec.loader.exec_module(mod)
    return mod


kj = _load(REPO / "scripts" / "kaggle_jobs.py")


def _all_commands():
    for job in kj.JOBS.values():
        for argv in job.shards + job.finalize:
            yield job.name, argv


@pytest.mark.parametrize("job,argv", list(_all_commands()), ids=lambda v: v if isinstance(v, str) else " ".join(v[:2]))
def test_every_command_parses(job, argv, monkeypatch, tmp_path):
    monkeypatch.setenv("KLTN_DATA_ROOT", str(tmp_path))
    script = REPO / argv[0]
    assert script.is_file(), f"{job}: {argv[0]} không tồn tại"
    if script.name == "kaggle_jobs.py":
        assert argv[1] in ("merge-e2e3",)
        return
    args = kj.render(argv)[1:]
    mod = _load(script)
    import inspect
    if inspect.signature(mod.parse_args).parameters:
        mod.parse_args(args)
    else:
        monkeypatch.setattr(sys, "argv", [script.name, *args])
        mod.parse_args()


def test_catalog_shape():
    names = set(kj.JOBS)
    for job in kj.JOBS.values():
        assert job.shards, job.name
        for dep in job.restore_from:
            assert dep in names or dep.startswith("("), (job.name, dep)
        if job.exclusive:
            assert job.parallel == 1
    timing = [j for j in kj.JOBS.values() if any("run_survey_e2_e3_timing.py" in a[0] for a in j.shards)]
    assert timing and all(j.exclusive for j in timing)


def test_run_parallel_reports_exit_codes(tmp_path):
    ok = [sys.executable, "-c", "print('hi')"]
    bad = [sys.executable, "-c", "import sys; sys.exit(4)"]
    assert kj.run_parallel([ok, bad, ok], 2, tmp_path, "t") == [0, 4, 0]
    assert (tmp_path / "t_0.log").read_text().strip() == "hi"


def test_merge_e2e3(tmp_path, monkeypatch):
    monkeypatch.setenv("KLTN_DATA_ROOT", str(tmp_path))
    # provenance thật đòi cây làm việc sạch và HEAD đã push; test chỉ kiểm tra logic gộp -> thay bằng bản ghi giả
    import survey_v2.runlog as runlog
    monkeypatch.setattr(runlog, "start_run", lambda script, params=None, data_files=(): {
        "commit_hash": "test-merge", "tracked": True, "traceable": True, "traceable_errors": [], "params": params or {}})
    root = tmp_path.joinpath("results", "survey_v2")
    ref = root / "E2_E3_ref"
    ref.mkdir(parents=True)
    pd.DataFrame({"method": ["f_test", "l1_lr", "relieff"], "wall_s": [0.1, 10.0, 800.0], "maxrss_isolated_mb": [5, 60, 100],
                  "peak_mb": [1, 2, 3], "maxrss_mb": [700, 700, 700]}).to_csv(ref / "e2_cost_point.csv", index=False)
    (ref / "machine_config.json").write_text("{}")
    new_fields = {"timing_tracemalloc": False, "peak_source": "VmHWM", "isolate_memory": True}     # v3.1.6 (QĐ27)
    (ref / "config.json").write_text(json.dumps({"traceable": True, "calibration_ok": None, "commit_hash": "abc", "methods": ["f_test", "l1_lr", "relieff"], **new_fields}))
    for p, m in [("E2_E3_p1", "f_test"), ("E2_E3_p2", "l1_lr"), ("E2_E3_p3", "relieff")]:
        d = root / p
        d.mkdir(parents=True)
        pd.DataFrame({"method": [m], "n": [100], "wall_s": [1.0]}).to_csv(d / "e3_scaling_n.csv", index=False)
        pd.DataFrame({"method": [m], "alpha_n": [1.0]}).to_csv(d / "e3_exponents.csv", index=False)
        (d / "e3_wrapper_evals.csv").write_bytes(b"\xef\xbb\xbf\n")       # tệp rỗng như các phiên 30/10 (chỉ BOM + xuống dòng)
        (d / "config.json").write_text(json.dumps({"traceable": True, "calibration_ok": True, "commit_hash": "abc", "methods": [m], "n_grid": [100], **new_fields}))
    out = kj.merge_e2e3()
    e2 = pd.read_csv(out / "e2_cost_point.csv")
    assert e2.method.tolist() == ["f_test", "l1_lr", "relieff"] and set(e2.session) == {"E2_E3_ref"}      # E2 chính thức = phiên tham chiếu
    assert "maxrss_isolated_mb" in e2.columns and "peak_mb" not in e2.columns and "maxrss_mb" not in e2.columns   # QĐ27
    assert not (out / "e3_wrapper_evals.csv").exists()                                                   # tệp rỗng bị bỏ qua, không lỗi
    e3 = pd.read_csv(out / "e3_scaling_n.csv")
    assert e3.method.tolist() == ["f_test", "l1_lr", "relieff"] and set(e3.session) == {"E2_E3_p1", "E2_E3_p2", "E2_E3_p3"}
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["sources_commit_hash"] == "abc" and cfg["calibration_ok"] is True and "commit_hash" in cfg   # stamp bằng provenance
    assert (out / "machine_config.json").exists()
    (root / "E2_E3_p2" / "config.json").write_text(json.dumps({"traceable": True, "calibration_ok": False, "commit_hash": "abc", **new_fields}))
    with pytest.raises(SystemExit):
        kj.merge_e2e3()
    # phiên đo ở bản trước v3.1.6 (tracemalloc bật / bộ nhớ ru_maxrss) -> không gộp (QĐ27)
    (root / "E2_E3_p2" / "config.json").write_text(json.dumps({"traceable": True, "calibration_ok": True, "commit_hash": "abc"}))
    with pytest.raises(SystemExit, match="tracemalloc"):
        kj.merge_e2e3()
    (root / "E2_E3_p2" / "config.json").write_text(json.dumps({"traceable": True, "calibration_ok": True, "commit_hash": "abc", **new_fields}))
    (ref / "config.json").write_text(json.dumps({"traceable": True, "calibration_ok": None, "commit_hash": "abc", "timing_tracemalloc": False,
                                                 "isolate_memory": True, "peak_source": "ru_maxrss"}))
    with pytest.raises(SystemExit, match="ru_maxrss"):
        kj.merge_e2e3()


def test_requires_and_redo_jobs():
    """v3.1.6: việc dùng kết quả pha A phải khai báo `requires` (E5c từng bị bỏ qua im lặng); việc chọn lại SVM-RFE có trong danh mục."""
    for name in ("e4e5_gse37745", "e4_tcga", "e1_phaseB_geo", "e1_phaseB_tcga", "ablation_phaseB", "tables", "redo_svm_rfe_A", "redo_svm_rfe_B"):
        assert kj.JOBS[name].requires, name
    for name in ("redo_svm_rfe_A", "redo_svm_rfe_B"):
        for argv in kj.JOBS[name].shards:
            assert "--redo-methods" in argv and argv[argv.index("--redo-methods") + 1] == "svm_rfe", (name, argv)
    assert all("--methods" in a for a in kj.JOBS["redo_svm_rfe_B"].shards)      # pha B/E1b/E4 chỉ chạy svm_rfe
    assert "redo_svm_rfe_A" in kj.JOBS["redo_svm_rfe_B"].restore_from


def test_restore_order_requires_cfg_and_panels(tmp_path, monkeypatch):
    """v3.1.7 (Review 02/11, QĐ30–QĐ32): restore_from = thứ tự giải nén (output mới nhất cuối); requires_cfg chặn khôi phục sai bản;
    việc `panels` chạy hai panel công bố qua pha A/B và E1b; `tables` chỉ chạy sau `panels`."""
    assert kj.JOBS["redo_svm_rfe_B"].restore_from[-1] == "redo_svm_rfe_A"
    assert kj.JOBS["ablation_phaseB"].restore_from[-1] == "redo_svm_rfe_A"
    assert kj.JOBS["tables"].restore_from[-2:] == ["redo_svm_rfe_B", "panels"]
    assert kj.JOBS["panels"].restore_from == ["redo_svm_rfe_B"]
    assert kj.PUBLISHED_PANELS == ["panel_girard2016", "panel_wilkerson2013"]
    pan = kj.JOBS["panels"]
    for argv in pan.shards + pan.finalize:
        assert "--redo-methods" in argv and set(kj.PUBLISHED_PANELS) <= set(argv), argv
        assert "svm_rfe" not in argv
    assert sum("run_e1_feature_selection" in a[0] for a in pan.shards) == 5
    assert sum("run_e1_classification" in a[0] for a in pan.finalize) == 5 and sum("e1b" in a[0] for a in pan.finalize) == 4
    assert all("--no-references" in a for a in pan.finalize if "run_e1_classification" in a[0])
    # requires_cfg: đúng bản -> không báo; sai bản (chưa chọn lại svm_rfe) -> báo và nêu thứ tự khôi phục
    root = tmp_path / "results" / "survey_v2"
    monkeypatch.setenv("KLTN_DATA_ROOT", str(tmp_path))
    job = kj.Job("t", "x", "x", [], [], requires=["E1_feature_selection/gse10245/selected_genes.csv"],
                 requires_cfg=[("E1_feature_selection/gse10245/config.json", "redo_methods", "svm_rfe"),
                               ("E2_E3/config.json", "timing_tracemalloc", False)], restore_from=["a", "b"])
    assert len(kj.check_requires(job)) == 3                                   # thiếu cả ba
    (root / "E1_feature_selection" / "gse10245").mkdir(parents=True)
    (root / "E1_feature_selection" / "gse10245" / "selected_genes.csv").write_text("x\n", encoding="utf-8")
    (root / "E1_feature_selection" / "gse10245" / "config.json").write_text(json.dumps({"redo_methods": []}), encoding="utf-8")
    (root / "E2_E3").mkdir()
    (root / "E2_E3" / "config.json").write_text(json.dumps({"timing_tracemalloc": True}), encoding="utf-8")
    msgs = kj.check_requires(job)
    assert len(msgs) == 2 and "redo_methods = []" in msgs[0] and "a, b" in msgs[0] and "timing_tracemalloc = True" in msgs[1]
    (root / "E1_feature_selection" / "gse10245" / "config.json").write_text(json.dumps({"redo_methods": ["svm_rfe"]}), encoding="utf-8")
    (root / "E2_E3" / "config.json").write_text(json.dumps({"timing_tracemalloc": False}), encoding="utf-8")
    assert kj.check_requires(job) == []
    for name in ("redo_svm_rfe_B", "ablation_phaseB", "tables", "panels"):
        assert kj.JOBS[name].requires_cfg, name


def test_pip_freeze_written(tmp_path):
    out = kj.write_pip_freeze(kj.JOBS["sanity"], tmp_path)
    assert out is not None and out.exists() and out.read_text(encoding="utf-8").startswith("# python ")


def test_run_parallel_stops_at_deadline(tmp_path):
    import time
    slow = [sys.executable, "-c", "import time; time.sleep(60)"]
    t0 = time.time()
    codes = kj.run_parallel([slow, slow, slow], 2, tmp_path, "slow", deadline=time.time() + 3)
    assert codes == [kj.TIMEOUT_CODE] * 3 and time.time() - t0 < 40


def test_runner_notebook_only_dispatches_to_kaggle_jobs():
    nb = json.loads((REPO / "notebooks" / "kaggle_survey_v3_runner.ipynb").read_text(encoding="utf-8"))
    code = "\n".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "scripts/kaggle_jobs.py" in code and "JOB = " in code
    assert "scripts/run_" not in code and "make_survey_tables" not in code      # M18: không có lệnh thí nghiệm trong notebook
    assert "KLTN_STRICT_N" in code and "'traceable'" not in code               # traceable do script kiểm, không gán tay


mk = _load(REPO / "scripts" / "make_kaggle_notebooks.py")


def test_per_experiment_notebooks_cover_every_job_once_and_are_up_to_date():
    covered = [n for e in mk.EXPERIMENTS for n in e.jobs]
    assert sorted(covered) == sorted(kj.JOBS) and len(covered) == len(set(covered))
    assert mk.main(["--check"]) == 0          # notebooks/kaggle/*.ipynb khớp runner + danh mục (sinh lại nếu lệch)


@pytest.mark.parametrize("exp", mk.EXPERIMENTS, ids=lambda e: e.file)
def test_per_experiment_notebook_only_dispatches(exp):
    nb = json.loads((REPO / "notebooks" / "kaggle" / exp.file).read_text(encoding="utf-8"))
    code = "\n".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert f"JOB = {exp.jobs[0]!r}" in code and "assert JOB in JOBS_IN_NOTEBOOK" in code
    assert "scripts/kaggle_jobs.py" in code and "scripts/run_" not in code and "make_survey_tables" not in code
    assert "KLTN_STRICT_N" in code and "'traceable'" not in code
