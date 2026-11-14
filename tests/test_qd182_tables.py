"""scripts/make_qd182_tables.py: QĐ182 đối chứng âm (Review 12/11 §5) trên dữ liệu giả. Viết và commit trước khi mở RQ2_d3_shuffled
(thứ tự mù QĐ69b)."""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
from rq2_fake_zip import COHORTS, _write_folder, cohort_data, patch_run, shuffled_data, write_rq2_zip, write_rq4_zip  # noqa: E402
from test_rq2_tables import synthetic  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("make_qd182_tables_under_test", REPO / "scripts" / "make_qd182_tables.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load()
PERM = {"source_labels_shuffled": True, "shuffle_seed": 7}


def _is_to(f):
    return (f.method == "mvp") & (f.gamma == 0) & (f.prefilter_space == "target")


def d3_copy_of_to(main: dict) -> dict:
    """Nhánh D3 xáo nhãn = bản chép đúng target-only của lần chính (điểm và AUROC): mọi Δ so với TO bằng 0."""
    out = {}
    for c, (per, sc) in main.items():
        to = per[_is_to(per) & (per.gamma_rule == "fixed") & per.n_t.astype(str).isin(m.N_T_MAIN)]
        ts = sc[_is_to(sc) & sc.n_t.astype(str).isin(m.N_T_MAIN)]
        p, s = [], []
        for arm in ("select_on_source", "pooled"):
            p.append(to.assign(method=arm, gamma=np.nan, gamma_rule="none", prefilter_space="source"))
            s.append(ts.assign(method=arm, gamma=np.nan, prefilter_space="source"))
        out[c] = (pd.concat(p, ignore_index=True), pd.concat(s, ignore_index=True))
    return out


def d3_random(main: dict, seed0: int = 300) -> dict:
    out = {}
    for i, (c, (per_main, _)) in enumerate(main.items()):
        n_ts = tuple(v for v in sorted(per_main.n_t.astype(str).unique(), key=int) if v in m.N_T_MAIN)
        per, sc = synthetic(cohort=c, n_ts=n_ts, ks=(2, 3, 4, 5), seed=seed0 + i, arms=("d3",))
        keep = per.method.isin(["select_on_source", "pooled"])
        out[c] = (per[keep].reset_index(drop=True), sc[sc.method.isin(["select_on_source", "pooled"])].reset_index(drop=True))
    return out


def write_d3_zip(path: Path, d3: dict, cfg: dict | None = None) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        _write_folder(zf, "RQ2_d3_shuffled", d3, cfg if cfg is not None else PERM, seed=500)
    return path


def test_permutation_must_match_mvp_shuffled():
    ok = {c: dict(PERM) for c in COHORTS}
    assert set(m.check_permutation(ok, ok)) == set(COHORTS)
    bad = {c: {**PERM, "shuffle_seed": 8} for c in COHORTS}
    with pytest.raises(SystemExit):
        m.check_permutation(ok, bad)
    with pytest.raises(SystemExit):
        m.check_permutation(ok, {c: {**PERM, "source_labels_shuffled": False} for c in COHORTS})


def test_cells_must_cover_target_only_runs():
    main = cohort_data()
    d3 = d3_copy_of_to(main)
    shuf = shuffled_data(main)
    assert set(m.check_cells(main, d3, shuf)) == set(COHORTS)
    per, sc = d3["gse30219"]
    d3["gse30219"] = (per.iloc[1:], sc)
    with pytest.raises(SystemExit):
        m.check_cells(main, d3, shuf)


def test_identical_arms_give_zero_delta_and_negative_labels():
    main = cohort_data()
    shuf = shuffled_data(main)                       # m5_cv xáo = TO của lần chính ở mọi lần chạy
    t = m.tables(m.merge(main, shuf, d3_copy_of_to(main)), n_boot=20)
    assert (t["cells"].delta == 0).all() and (t["cells"].se == 0).all()
    assert t["pooled"].ivw_undefined.all()
    lab = m.labels(t["pooled"]).set_index(["comparison", "n_t"]).label
    for comp in m.HARM:
        assert lab[(comp, "all")] == m.LABEL_HARM[False]
    for comp in m.LESS:
        assert lab[(comp, "all")] == m.LABEL_LESS[False]      # cận dưới = 0, không > 0


def test_delta_matches_mean_over_runs_of_k_average():
    main = cohort_data()
    shuf = shuffled_data(main)
    d3 = d3_random(main)
    cells = m.tables(m.merge(main, shuf, d3), n_boot=20)["cells"]
    r = cells[(cells.cohort == "gse37745") & (cells.n_t == "10") & (cells.comparison == "sos_shuffled_vs_target_only")].iloc[0]
    sel = lambda f, q: f.query(q).set_index(["split", "subset", "k"]).auroc  # noqa: E731
    a = sel(d3["gse37745"][0], "method == 'select_on_source' and n_t == '10' and k in [2, 3, 5]")
    b = sel(main["gse37745"][0], "gamma_rule == 'fixed' and gamma == 0 and prefilter_space == 'target' and n_t == '10' and k in [2, 3, 5]")
    assert r.delta == pytest.approx((a - b).groupby(level=["split", "subset"]).mean().mean(), abs=1e-12)


def test_harm_rule_uses_upper_bound_of_both_pools():
    pooled = pd.DataFrame([
        {"comparison": "sos_shuffled_vs_target_only", "pool_set": "gpl570", "n_t": "5", "cohorts": "a b", "delta_ivw": -0.05, "se_ivw": 0.01,
         "delta_eq": -0.05, "se_eq": 0.01, "lower_ivw": -0.07, "lower_eq": -0.07},
        {"comparison": "sos_shuffled_vs_target_only", "pool_set": "gpl570", "n_t": "10", "cohorts": "a b", "delta_ivw": -0.05, "se_ivw": 0.01,
         "delta_eq": -0.02, "se_eq": 0.01, "lower_ivw": -0.07, "lower_eq": -0.04},
        {"comparison": "m5_cv_shuffled_vs_sos_shuffled", "pool_set": "gpl570", "n_t": "5", "cohorts": "a b", "delta_ivw": 0.05, "se_ivw": 0.01,
         "delta_eq": 0.05, "se_eq": 0.01, "lower_ivw": 0.0336, "lower_eq": 0.0336}])
    harm_ivw_upper = -0.05 + m.Z95 * 0.01
    assert harm_ivw_upper < -0.01
    p = pooled.copy()
    p["ivw_undefined"] = False
    p["upper_ivw"] = p.delta_ivw + m.Z95 * p.se_ivw
    p["upper_eq"] = p.delta_eq + m.Z95 * p.se_eq
    harm = (p.upper_ivw < -m.MARGIN) & (p.upper_eq < -m.MARGIN)
    less = (p.lower_ivw > 0) & (p.lower_eq > 0)
    p["rule_met"] = np.where(p.comparison.isin(list(m.HARM)), harm, less)
    lab = m.labels(p).set_index(["comparison", "n_t"]).label
    assert lab[("sos_shuffled_vs_target_only", "5")] == m.LABEL_HARM[True]
    assert lab[("sos_shuffled_vs_target_only", "10")] == m.LABEL_HARM[False]          # bản trọng số bằng nhau: cận trên > −0,01
    assert lab[("sos_shuffled_vs_target_only", "all")] == m.LABEL_HARM[False]
    assert lab[("m5_cv_shuffled_vs_sos_shuffled", "5")] == m.LABEL_LESS[True]


def test_main_writes_tables_and_config(tmp_path, monkeypatch):
    main = cohort_data()
    rq2 = write_rq2_zip(tmp_path / "rq2_mvp_fake_results.zip", main)
    rq4 = write_rq4_zip(tmp_path / "rq4_h4b_noihc_fake_results.zip", main)
    d3 = write_d3_zip(tmp_path / "qd182_fake_results.zip", d3_random(main))
    patch_run(monkeypatch, m, tmp_path / "out")
    assert m.main(["--rq2", str(rq2), "--rq4", str(rq4), "--d3", str(d3), "--n-boot", "10"]) == 0
    out = tmp_path / "out" / "RQ2_qd182_tables"
    assert {p.stem for p in out.glob("*.csv")} == {"qd182_cells", "qd182_pooled", "qd182_gse42127", "qd182_labels"}
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["seed_key"] == "qd182" and len(cfg["input_sha256"]) == 3 and cfg["results_opened_at_utc"].endswith("UTC")
    assert set(cfg["permutation_check"]) == set(COHORTS)
    lab = pd.read_csv(out / "qd182_labels.csv", encoding="utf-8-sig")
    assert set(lab.label) <= set(m.LABEL_HARM.values()) | set(m.LABEL_LESS.values())
    assert set(pd.read_csv(out / "qd182_gse42127.csv", encoding="utf-8-sig").cohort) == {"gse42127"}
    d3_bad = write_d3_zip(tmp_path / "qd182_bad.zip", d3_random(main), cfg={**PERM, "shuffle_seed": 99})
    with pytest.raises(SystemExit):
        m.main(["--rq2", str(rq2), "--rq4", str(rq4), "--d3", str(d3_bad), "--n-boot", "10"])
