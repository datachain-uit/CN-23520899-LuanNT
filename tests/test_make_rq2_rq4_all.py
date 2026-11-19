"""scripts/make_rq2_rq4_all.py và scripts/make_rq2_rq4_figures.py (PG-04): một lệnh cho bảng và hình RQ2, RQ4."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


a = _load("rq2rq4_all_under_test", "make_rq2_rq4_all.py")
f = _load("rq2rq4_fig_under_test", "make_rq2_rq4_figures.py")


# ------------------------------------------------------------------------------------------------- make_rq2_rq4_all
def test_compare_frames_tolerance_nan_and_strings():
    x = pd.DataFrame({"n_t": ["5", "pool"], "delta": [0.1, np.nan], "note": ["a", None]})
    ok, _ = a.compare_frames(x, x.assign(delta=[0.1 + 5e-13, np.nan]))
    assert ok
    ok, d = a.compare_frames(x, x.assign(delta=[0.1 + 1e-9, np.nan]))
    assert not ok and "delta" in d
    ok, d = a.compare_frames(x, x.assign(delta=[0.1, 0.2]))
    assert not ok and "NaN" in d
    ok, d = a.compare_frames(x, x.assign(note=["b", None]))
    assert not ok and "note" in d
    ok, d = a.compare_frames(x, x[["delta", "n_t", "note"]])
    assert not ok


def test_check_folder_reports_missing_and_mismatched_files(tmp_path):
    folder = tmp_path / "RQ4_tables"
    folder.mkdir()
    pd.DataFrame({"n_t": ["5"], "v": [1.0]}).to_csv(folder / "same.csv", index=False)
    pd.DataFrame({"n_t": ["5"], "v": [2.0]}).to_csv(folder / "diff.csv", index=False)
    pd.DataFrame({"n_t": ["5"], "v": [1.0]}).to_csv(folder / "new_only.csv", index=False)
    z = tmp_path / "rq4_tables_abc_results.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("results/survey_v2/RQ4_tables/same.csv", "n_t,v\n5,1.0\n")
        zf.writestr("results/survey_v2/RQ4_tables/diff.csv", "n_t,v\n5,1.5\n")
        zf.writestr("results/survey_v2/RQ4_tables/ref_only.csv", "n_t,v\n5,1.0\n")
        zf.writestr("results/survey_v2/OTHER/same.csv", "n_t,v\n5,9\n")
    rows = {r["file"]: r for r in a.check_folder(folder, z)}
    assert rows["same.csv"]["match"] and not rows["diff.csv"]["match"]
    assert not rows["new_only.csv"]["match"] and not rows["ref_only.csv"]["match"]
    assert len(rows) == 4


def test_resolve_inputs_needs_exactly_one_zip_per_pattern(tmp_path):
    for n in ("rq2_mvp_1b7c021_results.zip", "rq4_h4b_noihc_d74b8c0_results.zip", "rq2_r_all_7f097da_results.zip",
              "qd182_d3_shuffled_c36c8c7_results.zip"):
        (tmp_path / n).write_bytes(b"")
    args = a.parse_args(["--zips", str(tmp_path)])
    inp = a.resolve_inputs(args)
    assert inp["rq2"].name == "rq2_mvp_1b7c021_results.zip" and inp["d3"].name.startswith("qd182")
    (tmp_path / "rq2_mvp_zzz_results.zip").write_bytes(b"")
    with pytest.raises(SystemExit):
        a.resolve_inputs(a.parse_args(["--zips", str(tmp_path)]))
    explicit = a.resolve_inputs(a.parse_args(["--zips", str(tmp_path), "--rq2", str(tmp_path / "rq2_mvp_zzz_results.zip")]))
    assert explicit["rq2"].name == "rq2_mvp_zzz_results.zip"


def test_steps_order_and_w1_reads_fresh_rq2_tables(tmp_path):
    inp = {k: tmp_path / f"{k}.zip" for k in a.INPUT_PATTERNS}
    s = a.steps(inp, tmp_path / "tables", 2000)
    names = [n for n, _ in s]
    assert names[0] == "make_rq2_tables.py" and names[1] == "make_w1_r_labels.py" and names[-1] == "make_rq2_rq4_figures.py"
    assert s[1][1] == ["--rq2-tables", str(tmp_path / "tables" / "RQ2_tables")]
    assert "--d3" in s[5][1] and len(a.steps(inp, tmp_path, 10, figures=False)) == 6
    for name, _ in s:
        assert (REPO / "scripts" / name).exists()


# ------------------------------------------------------------------------------------------------- make_rq2_rq4_figures
def _pooled(comparisons, nts=("5", "10", "20", "40", "pool")):
    rows = []
    for c in comparisons:
        for ps in ("gpl570", "all4"):
            for i, n in enumerate(nts):
                d = 0.01 * (i + 1)
                rows.append({"comparison": c, "n_t": n, "pool_set": ps, "delta_ivw": d, "delta_eq": d + 0.001,
                             "ci95_ivw_low": d - 0.01, "ci95_ivw_high": d + 0.01, "ci95_eq_low": d - 0.02, "ci95_eq_high": d + 0.02})
    return pd.DataFrame(rows)


def test_series_orders_n_t_and_keeps_primary_pool():
    df = _pooled(["target_only"], nts=("pool", "20", "5"))
    s = f.series(df, "target_only")
    assert s.n_t.tolist() == ["5", "20", "pool"] and (s.pool_set == "gpl570").all()
    with pytest.raises(ValueError):
        f.series(pd.concat([df, df]), "target_only")


def test_gamma_shares_sum_to_one_per_rule_and_n_t():
    g = pd.DataFrame({"cohort": ["a", "a", "b", "b"], "rule": "m5_pos", "n_t": "5", "k": [2, 2, 3, 3], "gamma": [0.5, 1.0, 0.5, 2.0],
                      "n_runs": [10, 10, 15, 5]})
    s = f.gamma_shares(g)
    assert s.share.sum() == pytest.approx(1.0)
    assert s.set_index("gamma").share[0.5] == pytest.approx(25 / 40)


def test_build_writes_all_figures(tmp_path):
    t = {"rq2_pooled": _pooled(list(f.LABEL)), "rq2_h2b": _pooled(["ihc_2", "ihc_4"], nts=("5", "10")).iloc[:0],
         "decomp": _pooled(list(f.DECOMP)), "h4a": _pooled(["mvp_vs_target_only", "mvp_vs_pooled"]),
         "h4b": _pooled(["h4b_mvp_shuffled_vs_target_only"])}
    t["rq2_cells"] = pd.DataFrame({"cohort": ["gse10245"] * 3 + ["gse42127"] * 3, "comparison": "target_only",
                                   "n_t": ["5", "10", "pool"] * 2, "delta": 0.01, "se": 0.005})
    t["gamma"] = pd.DataFrame({"cohort": "a", "rule": ["m5_pos", "m5_cv"], "n_t": "5", "k": 2, "gamma": [0.5, 0.0], "n_runs": 20})
    man = f.build(t, tmp_path)
    assert sorted(man.figure) == sorted(f.FIGURES)
    for name in f.FIGURES:
        assert (tmp_path / name).read_bytes()[:4] == b"%PDF"
