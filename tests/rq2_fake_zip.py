"""Zip giả nhỏ đúng bố cục của JOB rq2_mvp và rq4_h4b_noihc, để test gọi main() của các script bảng (Y90, Review 11/11).

Không phải tệp test (pytest không thu thập); dùng chung cho tests/test_rq4_tables.py, test_qd97_tables.py, test_qd126_tables.py.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_rq2_tables import synthetic  # noqa: E402

COHORTS = ("gse10245", "gse30219", "gse37745", "gse42127")
N_GENES = 40
PREFIX = "results/survey_v2/"


def genes_for(per: pd.DataFrame, seed: int = 0, stable_mvp: bool = True) -> pd.DataFrame:
    """Gen cho mọi dòng fixed (MVP γ > 0 gần cùng một panel; γ = 0 ngẫu nhiên) và cho D3 (select-on-source cố định, pooled ngẫu nhiên)."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in per[(per.gamma_rule == "fixed") | per.method.isin(["select_on_source", "pooled"])].itertuples():
        if r.method == "select_on_source" or (r.method == "mvp" and r.gamma > 0 and stable_mvp):
            genes = [f"G{j}" for j in range(r.k)]
        else:
            genes = [f"G{j}" for j in rng.choice(N_GENES, size=r.k, replace=False)]
        rows += [{"cohort": r.cohort, "split": r.split, "subset": r.subset, "n_t": r.n_t, "k": r.k, "method": r.method, "gamma": r.gamma,
                  "prefilter_space": r.prefilter_space, "rank": i + 1, "gene": g} for i, g in enumerate(genes)]
    return pd.DataFrame(rows)


def cohort_data(seed0: int = 10, shift: float = 0.6, arms=("d3", "ihc"), ks=(2, 3, 4, 5), chosen=None) -> dict:
    return {c: synthetic(cohort=c, n_ts=("5", "10") if c == "gse10245" else ("5", "10", "20"), ks=ks, seed=seed0 + i, mvp_shift=shift,
                         arms=arms, chosen=(chosen or {}).get(c))
            for i, c in enumerate(COHORTS)}


def _csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _write_folder(zf: zipfile.ZipFile, folder: str, data: dict, cfg: dict, seed: int) -> None:
    for i, (c, (per, sc)) in enumerate(data.items()):
        base = f"{PREFIX}{folder}/{c}/"
        zf.writestr(base + "rq2_mvp_per_run.csv", _csv(per))
        zf.writestr(base + "rq2_mvp_scores.csv", _csv(sc))
        zf.writestr(base + "rq2_mvp_selected_genes.csv", _csv(genes_for(per, seed=seed + i)))
        zf.writestr(base + "config.json", json.dumps({"gene_space_size": N_GENES, **cfg}))


def write_rq2_zip(path: Path, data: dict | None = None) -> Path:
    data = data or cohort_data()
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(PREFIX + "RQ2_mvp/rq2_equality_report.csv", _csv(pd.DataFrame({"status": ["ok", "ok"]})))
        _write_folder(zf, "RQ2_mvp", data, {}, seed=0)
    return path


def shuffled_data(main: dict, seed0: int = 70, chosen_cv=None) -> dict:
    """Biến thể nhãn nguồn xáo: chỉ nhánh MVP. TO (γ = 0 / target) chép từ lần chạy chính như JOB thật (cùng không gian gen, cùng seed);
    dòng m5_cv lấy γ theo chosen_cv[(split, subset, n_t, k)] (mặc định γ = 0, tức trùng TO)."""
    out = {}
    for i, (c, (per_main, sc_main)) in enumerate(main.items()):
        per, sc = synthetic(cohort=c, n_ts=tuple(sorted(per_main.n_t.unique(), key=int)), ks=tuple(sorted(per_main[per_main.method == "mvp"].k.unique())),
                            seed=seed0 + i,
                            mvp_shift=0.0, arms=())
        is_to = lambda f: (f.method == "mvp") & (f.gamma == 0) & (f.prefilter_space == "target")  # noqa: E731
        per = pd.concat([per[~(is_to(per) & (per.gamma_rule == "fixed"))], per_main[is_to(per_main) & (per_main.gamma_rule == "fixed")]],
                        ignore_index=True)
        sc = pd.concat([sc[~is_to(sc)], sc_main[is_to(sc_main)]], ignore_index=True)
        fixed = per[per.gamma_rule == "fixed"].set_index(["split", "subset", "n_t", "k", "gamma", "prefilter_space"])
        cv = []
        for r in per[per.gamma_rule == "m5_cv"].itertuples():
            g = (chosen_cv or {}).get((r.split, r.subset, r.n_t, r.k), 0.0)
            row = fixed.loc[(r.split, r.subset, r.n_t, r.k, g, "target" if g == 0 else "source")].to_dict()
            cv.append({"split": r.split, "subset": r.subset, "n_t": r.n_t, "k": r.k, "gamma": g,
                       "prefilter_space": "target" if g == 0 else "source", **row, "gamma_rule": "m5_cv"})
        per = pd.concat([per[per.gamma_rule != "m5_cv"], pd.DataFrame(cv)[per.columns]], ignore_index=True)
        out[c] = (per, sc)
    return out


def write_rq4_zip(path: Path, main: dict, shuffled: dict | None = None, noihc: dict | None = None) -> Path:
    shuffled = shuffled or shuffled_data(main)
    noihc = noihc or cohort_data(seed0=40, shift=0.0, arms=())
    with zipfile.ZipFile(path, "w") as zf:
        _write_folder(zf, "RQ2_mvp_shuffled", shuffled, {"source_labels_shuffled": True, "shuffle_seed": 7}, seed=100)
        _write_folder(zf, "RQ2_mvp_noIHC", noihc, {"excluded_genes": ["NAPSA", "NKX2-1", "KRT5", "TP63"]}, seed=200)
    return path


def fake_prov() -> dict:
    return {"commit_hash": "test000", "data_sha256": {}, "tracked": False, "traceable": False, "traceable_errors": []}


def patch_run(monkeypatch, module, out_root: Path) -> None:
    """main() không cần git: start_run trả provenance giả, results_dir ghi vào out_root."""
    monkeypatch.setattr(module, "start_run", lambda script, params=None, data_files=(): fake_prov())

    def results_dir(*parts):
        d = Path(out_root).joinpath(*parts)
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr(module, "results_dir", results_dir)
