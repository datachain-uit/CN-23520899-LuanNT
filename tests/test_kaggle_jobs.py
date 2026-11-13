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


def test_tables_requires_cfg_accepts_panels_output(tmp_path, monkeypatch):
    """v3.1.7a: `tables` chạy sau `panels` — config.json của E1_feature_selection khi đó có redo_methods = hai panel.
    Điều kiện của `tables` phải đạt với đúng output của `panels` và vẫn chặn khi bản cũ (chưa có panel) ghi đè."""
    monkeypatch.setenv("KLTN_DATA_ROOT", str(tmp_path))
    root = tmp_path / "results" / "survey_v2"
    job = kj.JOBS["tables"]
    fs_rules = [r for r in job.requires_cfg if r[0].startswith("E1_feature_selection/")]
    assert fs_rules and all(v in kj.PUBLISHED_PANELS for _, _, v in fs_rules)
    for redo, ok in ((list(kj.PUBLISHED_PANELS), True), (["svm_rfe"], False)):
        for d in kj.C.DATASETS:
            (root / "E1_feature_selection" / d).mkdir(parents=True, exist_ok=True)
            (root / "E1_feature_selection" / d / "config.json").write_text(json.dumps({"redo_methods": redo}), encoding="utf-8")
        msgs = [m for m in kj.check_requires(job) if m.startswith("E1_feature_selection/")]
        assert (msgs == []) is ok, msgs
    # đầu vào của `panels` vẫn phải là bản svm_rfe đã chọn lại
    assert all(v == "svm_rfe" for r, k, v in kj.JOBS["panels"].requires_cfg if r.startswith("E1_feature_selection/"))


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


def _config_source(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "".join(nb["cells"][1]["source"])


def test_rq2_early_job_uses_script_defaults_and_needs_no_restore():
    """Bước 8 (Review 03/11): chỉ cần dữ liệu TCGA; k, n_T lấy từ mặc định của script (M18), không ghi trong danh mục."""
    job = kj.JOBS["rq2_early"]
    assert job.outputs == ["RQ2_mvp_early/tcga_early"] and job.restore_from == [] and job.requires == [] and not job.exclusive
    assert len(job.shards) == 1 and "--k" not in job.shards[0] and "--n-t" not in job.shards[0]
    args = _load(REPO / job.shards[0][0]).parse_args(job.shards[0][1:])
    assert args.early_tcga and args.save_scores and args.out == "RQ2_mvp_early"
    assert args.k == [2, 4, 5, 10] and args.n_t == ["5", "10"] and args.gamma_rule == "cv" and args.arms == ["mvp"]


def test_rq2_notebook_clones_rq2_branch_and_survey_notebooks_stay_on_survey_branch():
    """Code MVP ở nhánh rq2-mvp của repo cá nhân; notebook survey phải giữ repo và nhánh cũ để tái lập số liệu đã chốt."""
    for name in ("kaggle_rq2_early.ipynb", "kaggle_rq2_mvp.ipynb", "kaggle_rq1_sensitivity_b.ipynb", "kaggle_rq4_h4b_noihc.ipynb", "kaggle_qd182_d3_shuffled.ipynb"):
        rq2 = _config_source(REPO / "notebooks" / "kaggle" / name)
        assert "REPO_BRANCH = 'rq2-mvp'" in rq2 and "REPO_URL = 'https://github.com/datachain-uit/CN-23520899-LuanNT.git'" in rq2
    for exp in mk.EXPERIMENTS:
        if exp.file not in ("kaggle_rq2_early.ipynb", "kaggle_rq2_mvp.ipynb", "kaggle_rq1_sensitivity_b.ipynb", "kaggle_rq4_h4b_noihc.ipynb", "kaggle_qd182_d3_shuffled.ipynb"):
            other = _config_source(REPO / "notebooks" / "kaggle" / exp.file)
            assert "REPO_BRANCH = 'survey-v2-code'" in other and "REPO_URL = 'https://github.com/datachain-uit/RNA-Seq-FS-Survey.git'" in other, exp.file
    assert "REPO_BRANCH = 'survey-v2-code'" in _config_source(REPO / "notebooks" / "kaggle_survey_v3_runner.ipynb")


def test_rq2_mvp_job_runs_all_arms_and_gates_on_equality():
    """Review 07/11 Bảng 7 bước 5, 6: đủ 4 cohort, 5 mức k, 4 γ, có điểm từng mẫu; khôi phục E7 rồi panels (E1b mới nhất); cổng nằm ở bước verify."""
    job = kj.JOBS["rq2_mvp"]
    assert job.outputs == [f"RQ2_mvp/{d}" for d in kj.GEO] and job.restore_from == ["e7", "panels"]
    assert set(job.requires) == {f"E1b/{d}/e1b_per_run.csv" for d in kj.GEO} | {f"E7/{d}/e7_per_run.csv" for d in kj.GEO}
    assert len(job.shards) == 1
    args = _load(REPO / job.shards[0][0]).parse_args(job.shards[0][1:])
    assert args.datasets == list(kj.GEO) and args.k == [2, 3, 4, 5, 10] and args.gamma == [0.0, 0.5, 1.0, 2.0]
    assert args.save_scores and args.out == "RQ2_mvp" and args.gamma_rule == "cv" and args.arms == ["mvp", "d3", "ihc"] and args.n_t is None
    assert [g[0] for g in job.gates] == ["scripts/check_rq2_equality.py"]


@pytest.mark.parametrize("code, passes", [(0, True), (3, False)])
def test_verify_reports_failing_gate(tmp_path, monkeypatch, code, passes):
    monkeypatch.setenv("KLTN_DATA_ROOT", str(tmp_path))
    job = kj.Job("x", "b", "w", shards=[], outputs=[], gates=[["-c", f"import sys; sys.exit({code})"]])
    problems = kj.verify(job, None)
    assert (problems == []) is passes
    assert passes or "mã 3" in problems[0]


@pytest.mark.parametrize("path", ["notebooks/kaggle_survey_v3_runner.ipynb", "notebooks/kaggle/kaggle_rq2_mvp.ipynb"])
def test_runner_archive_name_has_commit_and_still_matches_restore_glob(path):
    """Y58: tên zip kèm commit để tải lại không ghi đè; ô khôi phục tìm theo *_results.zip nên vẫn nhận tên mới."""
    import fnmatch

    nb = json.loads((REPO / path).read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "f'{JOB}_{commit_hash[:7]}_results.zip'" in code
    assert "rglob('*_results.zip')" in code and fnmatch.fnmatch("rq2_mvp_abc1234_results.zip", "*_results.zip")


# ------------------------------------------------------------------ QĐ70 (Review 08/11): kiểm phiên bản trước khi chạy rq2_mvp
REF_VERSIONS = {"numpy": "2.1.3", "pandas": "2.3.3", "scipy": "1.16.3", "scikit-learn": "1.6.1"}


def _write_ref_configs(root: Path, versions_by_rel: dict):
    for rel, versions in versions_by_rel.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"traceable": True, "versions": versions}), encoding="utf-8")


def test_rq2_mvp_job_pins_versions_of_e1b_and_e7():
    job = kj.JOBS["rq2_mvp"]
    assert set(job.pin_versions) == {f"E1b/{d}/config.json" for d in kj.GEO} | {f"E7/{d}/config.json" for d in kj.GEO}
    assert set(kj.PINNED_PACKAGES) == set(REF_VERSIONS)


def test_check_versions_passes_when_session_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("KLTN_DATA_ROOT", str(tmp_path))
    job = kj.Job("x", "b", "w", shards=[], outputs=[], pin_versions=["E1b/a/config.json", "E7/a/config.json"])
    _write_ref_configs(kj.results_root(), {r: dict(REF_VERSIONS, tables="3.9") for r in job.pin_versions})
    assert kj.check_versions(job, current=dict(REF_VERSIONS)) == ([], {})


def test_check_versions_reports_mismatch_and_pip_pins(tmp_path, monkeypatch):
    monkeypatch.setenv("KLTN_DATA_ROOT", str(tmp_path))
    job = kj.Job("x", "b", "w", shards=[], outputs=[], pin_versions=["E1b/a/config.json"])
    _write_ref_configs(kj.results_root(), {"E1b/a/config.json": REF_VERSIONS})
    problems, pins = kj.check_versions(job, current=dict(REF_VERSIONS, **{"scikit-learn": "1.7.0"}))
    assert len(problems) == 1 and "scikit-learn" in problems[0] and "1.7.0" in problems[0] and "1.6.1" in problems[0]
    assert pins == {"scikit-learn": "1.6.1"}


@pytest.mark.parametrize("case", ["missing_cfg", "missing_field", "refs_disagree"])
def test_check_versions_refuses_unusable_references(tmp_path, monkeypatch, case):
    monkeypatch.setenv("KLTN_DATA_ROOT", str(tmp_path))
    job = kj.Job("x", "b", "w", shards=[], outputs=[], pin_versions=["E1b/a/config.json", "E7/a/config.json"])
    refs = {"E1b/a/config.json": REF_VERSIONS, "E7/a/config.json": REF_VERSIONS}
    if case == "missing_cfg":
        refs.pop("E7/a/config.json")
    elif case == "missing_field":
        refs["E7/a/config.json"] = {k: v for k, v in REF_VERSIONS.items() if k != "scipy"}
    else:
        refs["E7/a/config.json"] = dict(REF_VERSIONS, numpy="2.0.0")
    _write_ref_configs(kj.results_root(), refs)
    problems, pins = kj.check_versions(job, current=dict(REF_VERSIONS))
    assert problems and pins == {}


def test_cmd_run_stops_with_code_2_on_version_mismatch(tmp_path, monkeypatch):
    """Bảng 7 bước 1: config giả lệch phiên bản -> mã 2, không chạy shard nào; trùng -> chạy."""
    monkeypatch.setenv("KLTN_DATA_ROOT", str(tmp_path))
    marker = tmp_path / "ran.txt"
    job = kj.Job("x", "b", "w", shards=[["-c", f"open(r'{marker}', 'w').write('1')"]], outputs=[],
                 pin_versions=["E1b/a/config.json"], parallel=1)
    monkeypatch.setattr(kj, "write_pip_freeze", lambda *a, **k: None)
    monkeypatch.setattr(kj, "current_versions", lambda: dict(REF_VERSIONS, numpy="2.2.0"))
    _write_ref_configs(kj.results_root(), {"E1b/a/config.json": REF_VERSIONS})
    assert kj.cmd_run(job, None, tmp_path / "logs") == 2 and not marker.exists()
    monkeypatch.setattr(kj, "current_versions", lambda: dict(REF_VERSIONS))
    monkeypatch.setattr(kj.time, "sleep", lambda s: None)
    assert kj.cmd_run(job, None, tmp_path / "logs") == 0 and marker.exists()


def _code_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_rq2_mvp_notebook_has_pip_pins_installed_before_any_import():
    """QĐ70: PIP_PINS ở ô cấu hình (mặc định rỗng); Gate 2 cài đúng ==phiên bản trước khi import numpy/pandas/scipy/sklearn."""
    cells = _code_cells(REPO / "notebooks" / "kaggle" / "kaggle_rq2_mvp.ipynb")
    assert "\nPIP_PINS = {}" in cells[0]
    gate2 = next(c for c in cells if c.startswith("# Gate 2"))
    install, imports = gate2.index("if PIP_PINS:"), gate2.index("for mod in (")
    assert install < imports and "f'{p}=={v}'" in gate2[install:imports] and "check=True" in gate2[install:imports]
    assert not any("import numpy" in c or "import pandas" in c for c in cells[:2])


@pytest.mark.parametrize("exp", [e for e in mk.EXPERIMENTS if e.file != "kaggle_rq2_mvp.ipynb"], ids=lambda e: e.file)
def test_other_notebooks_have_no_pip_pins(exp):
    assert not any("PIP_PINS" in c for c in _code_cells(REPO / "notebooks" / "kaggle" / exp.file))


def test_version_mismatch_prints_pip_pins_line_to_paste(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KLTN_DATA_ROOT", str(tmp_path))
    job = kj.Job("x", "b", "w", shards=[["-c", "pass"]], outputs=[], pin_versions=["E1b/a/config.json"])
    monkeypatch.setattr(kj, "current_versions", lambda: dict(REF_VERSIONS, numpy="2.2.0", scipy="1.17.0"))
    _write_ref_configs(kj.results_root(), {"E1b/a/config.json": REF_VERSIONS})
    assert kj.cmd_run(job, None, tmp_path / "logs") == 2
    assert "PIP_PINS = {'numpy': '2.1.3', 'scipy': '1.16.3'}" in capsys.readouterr().out


def test_rq4_h4b_noihc_job_runs_two_mvp_variants_on_the_rq2_grid():
    job = kj.JOBS["rq4_h4b_noihc"]
    parsed = [_load(REPO / s[0]).parse_args(s[1:]) for s in job.shards]
    assert len(parsed) == 2 and all(a.arms == ["mvp"] and a.datasets == list(kj.GEO) and a.k == [2, 3, 4, 5, 10] for a in parsed)
    assert all(a.gamma == [0.0, 0.5, 1.0, 2.0] and a.save_scores and a.gamma_rule == "cv" for a in parsed)
    shuf, noihc = parsed
    assert shuf.shuffle_source_labels and shuf.out == "RQ2_mvp_shuffled" and shuf.exclude_genes is None
    assert noihc.exclude_genes == list(kj.IHC_GENES) and noihc.out == "RQ2_mvp_noIHC" and not noihc.shuffle_source_labels
    assert job.restore_from == [] and job.requires == [] and job.parallel == 2


def test_qd182_job_runs_d3_with_the_rq4_shuffle_on_the_e7_grid():
    job = kj.JOBS["qd182_d3_shuffled"]
    assert len(job.shards) == 1
    a = _load(REPO / job.shards[0][0]).parse_args(job.shards[0][1:])
    assert a.arms == ["d3"] and a.shuffle_source_labels and a.exclude_genes is None and a.out == "RQ2_d3_shuffled"
    assert a.datasets == list(kj.GEO) and a.k == [2, 3, 4, 5, 10] and a.n_t == ["5", "10", "20"] and a.save_scores
    rq4 = _load(REPO / kj.JOBS["rq4_h4b_noihc"].shards[0][0]).parse_args(kj.JOBS["rq4_h4b_noihc"].shards[0][1:])
    assert rq4.shuffle_source_labels and rq4.datasets == a.datasets and rq4.source == a.source   # cùng nguồn, cùng hoán vị mỗi cohort
    assert job.outputs == [f"RQ2_d3_shuffled/{d}" for d in kj.GEO] and job.restore_from == [] and job.requires == []
