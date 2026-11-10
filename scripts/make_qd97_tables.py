"""Bảng bổ sung bắt buộc theo QĐ97 (Review 09/11, Mục 6.4) từ đầu ra JOB rq2_mvp (và rq4_h4b_noihc cho QĐ97b); không chạy lại mô hình,
không đổi quy tắc hay bảng H2, H2b, H2c đã sinh (QĐ97f). Mọi bảng ở đây là MÔ TẢ (không Holm, không quyết định giả thuyết).

(a) Phân rã MVP − target-only trên cùng test_T, bootstrap hai tầng của make_rq2_tables.py, trung bình k ∈ {2, 3, 5}:
    Δ_space = TO_nguồn − TO; Δ_π = MVP − TO_nguồn; Δ_total = MVP − TO = Δ_space + Δ_π; Δ_D3 = MVP − select-on-source và MVP − pooled
    (n_T ≤ 20, nơi D3 có chạy). TO_nguồn = γ = 0 trên không gian gen của nguồn.
(b) Δ MVP(nhãn nguồn thật) − MVP(nhãn nguồn xáo) trên cùng test_T (RQ2_mvp vs RQ2_mvp_shuffled), cùng bootstrap.
(d) Bảng từng cohort: đúng các ô của make_rq2_tables.build (cùng seed, nên Δ và SE trùng rq2_cells.csv), thêm CI 95 % hai phía, rồi các
    dòng gộp (IVW và trọng số bằng nhau, I², trọng số từng cohort). Chỉ số phụ BAcc, MCC, log-loss, Brier (và AUROC để đối chiếu) cho mọi
    nhánh: trung bình theo lần chạy (mỗi lần chạy trung bình k ∈ {2, 3, 5}), hiệu ghép cặp MVP − nhánh, SD và SE = SD/√n theo lần chạy
    (chỉ tầng lần chạy; log-loss, Brier: âm = MVP tốt hơn); dòng gộp là trung bình bằng nhau các cohort. Bảng riêng GSE42127 theo n_T và
    từng k ∈ {2, 3, 4, 5, 10} (và trung bình 2 3 5): MVP − target-only, − select-on-source, − pooled, Δ_space, Δ_π.

    python scripts/make_qd97_tables.py --rq2 rq2_mvp_1b7c021_results.zip --rq4 rq4_h4b_noihc_d74b8c0_results.zip \
        --rq2-tables rq2_tables_198b840_results.zip

--rq2-tables (tuỳ chọn): kiểm các ô từng cohort trùng rq2_cells.csv đã nộp (dừng nếu lệch). Đầu ra: results/survey_v2/RQ2_qd97_tables/
{qd97a_decomposition_cells.csv, qd97a_decomposition_pooled.csv, qd97b_true_vs_shuffled_cells.csv, qd97b_true_vs_shuffled_pooled.csv,
qd97d_cohort_table.csv, qd97d_secondary_metrics.csv, qd97d_gse42127_by_k.csv, config.json}.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import sha256  # noqa: E402
from survey_v2.runlog import all_versions, results_dir, start_run, write_config  # noqa: E402


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


m2 = _load("make_rq2_tables_for_qd97", "make_rq2_tables.py")
m4 = _load("make_rq4_tables_for_qd97", "make_rq4_tables.py")
K_AVG = m2.K_AVG
Z975 = m2.Z975
D3_N_T = tuple(m2.D3_N_T)
TO_SOURCE = m4.TO_SOURCE
DESCRIPTIVE = "mô tả (QĐ97; không kiểm định)"
DECOMP = [  # (tên, nhánh a, nhánh b, mức n_T được phép): Δ = a − b
    ("decomp_total", "mvp", "target_only", None),
    ("decomp_space", TO_SOURCE, "target_only", None),
    ("decomp_pi", "mvp", TO_SOURCE, None),
    ("decomp_d3_select_on_source", "mvp", "select_on_source", D3_N_T),
    ("decomp_d3_pooled", "mvp", "pooled", D3_N_T),
]
BY_K = [("target_only", "mvp", "target_only", None), ("select_on_source", "mvp", "select_on_source", D3_N_T),
        ("pooled", "mvp", "pooled", D3_N_T), ("decomp_space", TO_SOURCE, "target_only", None), ("decomp_pi", "mvp", TO_SOURCE, None)]
SEC_ARMS = ("target_only", TO_SOURCE, "select_on_source", "pooled")
SEC_METRICS = ("auroc", "bacc", "mcc", "logloss", "brier")
SHUFFLED = "mvp_shuffled"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq2", type=Path, required=True, help="zip của JOB rq2_mvp (hoặc thư mục RQ2_mvp)")
    ap.add_argument("--rq4", type=Path, required=True, help="zip của JOB rq4_h4b_noihc (dùng RQ2_mvp_shuffled)")
    ap.add_argument("--rq2-tables", type=Path, default=None, help="zip bảng RQ2 đã nộp (rq2_cells.csv) để kiểm trùng")
    ap.add_argument("--cohorts", nargs="+", default=list(C.FEWSHOT_COHORTS))
    ap.add_argument("--n-boot", type=int, default=C.BOOTSTRAP_N)
    ap.add_argument("--out", default="RQ2_qd97_tables")
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------------------------- so sánh ghép cặp
def _levels(per, allowed):
    lv = sorted(per.n_t.astype(str).unique(), key=m2._n_t_key)
    return lv if allowed is None else [v for v in lv if v in allowed]


def _cell(per, sc, cohort, n_t, name, a, b, ks, n_boot, tag):
    runs = m2.collect_runs(per, sc, cohort, n_t, a, b, ks)
    cell = m2.bootstrap_cell(runs, n_boot, m2.cell_seed(tag, cohort, n_t, name, *ks))
    return {"cohort": cohort, "comparison": name, "arm_a": a, "arm_b": b, "n_t": n_t, "k": " ".join(map(str, ks)), **cell,
            "ci95_low": cell["delta"] - Z975 * cell["se"], "ci95_high": cell["delta"] + Z975 * cell["se"],
            "auroc_a_mean": float(np.mean([np.mean(list(r.auroc_a.values())) for r in runs])),
            "auroc_b_mean": float(np.mean([np.mean(list(r.auroc_b.values())) for r in runs]))}


def _pool_rows(cells: pd.DataFrame, cohorts) -> pd.DataFrame:
    out = []
    sets = {**m2.POOL_SETS, "all4": tuple(cohorts)}
    for (name, n_t), g in cells.groupby(["comparison", "n_t"], sort=False):
        for set_name, members in sets.items():
            sub = g[g.cohort.isin(members)]
            if len(sub):
                out.append({"comparison": name, "n_t": n_t, "pool_set": set_name, "family": DESCRIPTIVE, "cohorts": " ".join(sub.cohort),
                            **m2.pool(sub, 0.0)})
    return pd.DataFrame(out)


def paired(data: dict, specs, n_boot: int, tag: str, ks=K_AVG) -> dict:
    cells = [_cell(per, sc, c, n_t, name, a, b, ks, n_boot, tag)
             for c, (per, sc) in data.items() for name, a, b, allowed in specs for n_t in _levels(per, allowed)]
    cells = pd.DataFrame(cells)
    return {"cells": cells, "pooled": _pool_rows(cells, data)}


def decomposition(data: dict, n_boot: int) -> dict:
    return paired(data, DECOMP, n_boot, tag="qd97a")


def true_vs_shuffled(data: dict, shuffled: dict, n_boot: int) -> dict:
    merged = {}
    for c, (per, sc) in data.items():
        sp, ss = m4.relabel(*shuffled[c], SHUFFLED)
        sp, ss = sp[sp.method == SHUFFLED], ss[ss.method == SHUFFLED]          # nhánh target-only của lần xáo trùng hệt bản chính
        merged[c] = (pd.concat([per, sp], ignore_index=True), pd.concat([sc, ss], ignore_index=True))
    return paired(merged, [("mvp_true_minus_shuffled", "mvp", f"{SHUFFLED}|{m2.MVP_RULE}||", None)], n_boot, tag="qd97b")


def cohort_by_k(data: dict, n_boot: int, cohort: str = "gse42127") -> pd.DataFrame:
    per, sc = data[cohort]
    ks_all = sorted(int(k) for k in per[per.method == "mvp"].k.unique())
    rows = []
    for ks in [(k,) for k in ks_all] + [K_AVG]:
        for name, a, b, allowed in BY_K:
            for n_t in _levels(per, allowed):
                rows.append(_cell(per, sc, cohort, n_t, name, a, b, ks, n_boot, "qd97d_k"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- bảng từng cohort, chỉ số phụ
def cohort_table(data: dict, n_boot: int) -> pd.DataFrame:
    built = m2.build(data, n_boot)
    cells = built["cells"].copy()
    cells.insert(0, "level", "cohort")
    cells["ci95_low"] = cells.delta - Z975 * cells.se
    cells["ci95_high"] = cells.delta + Z975 * cells.se
    pooled = built["pooled"].copy()
    pooled.insert(0, "level", "pooled")
    return pd.concat([cells, pooled], ignore_index=True)


def _run_means(per: pd.DataFrame, arm: str, n_t: str) -> pd.DataFrame:
    rows = m2._arm_rows(per, arm, n_t, K_AVG, m2.MVP_RULE)
    if rows.groupby(m2.RUN).k.nunique().min() != len(K_AVG):
        raise ValueError(f"{arm}, n_T = {n_t}: có lần chạy thiếu k")
    return rows.groupby(m2.RUN)[list(SEC_METRICS)].mean()


def secondary_metrics(data: dict) -> pd.DataFrame:
    rows = []
    for cohort, (per, sc) in data.items():
        per, _ = m2._normalize(per, sc.iloc[:0])
        for arm in SEC_ARMS:
            for n_t in _levels(per, D3_N_T if arm in ("select_on_source", "pooled") else None):
                a, b = _run_means(per, "mvp", n_t), _run_means(per, arm, n_t)
                if not a.index.equals(b.index):
                    a, b = a.align(b, join="inner", axis=0)
                for metric in SEC_METRICS:
                    d = a[metric] - b[metric]
                    rows.append({"level": "cohort", "cohort": cohort, "n_t": n_t, "arm": arm, "metric": metric, "n_runs": int(len(d)),
                                 "mvp_mean": float(a[metric].mean()), "arm_mean": float(b[metric].mean()), "delta_mean": float(d.mean()),
                                 "delta_sd": float(d.std(ddof=1)) if len(d) > 1 else np.nan,
                                 "delta_se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan})
    cells = pd.DataFrame(rows)
    sets = {**m2.POOL_SETS, "all4": tuple(data)}
    pooled = []
    for (n_t, arm, metric), g in cells.groupby(["n_t", "arm", "metric"], sort=False):
        for name, members in sets.items():
            sub = g[g.cohort.isin(members)]
            if len(sub):
                pooled.append({"level": "pooled", "cohort": name, "n_t": n_t, "arm": arm, "metric": metric, "n_runs": int(sub.n_runs.sum()),
                               "mvp_mean": float(sub.mvp_mean.mean()), "arm_mean": float(sub.arm_mean.mean()), "delta_mean": float(sub.delta_mean.mean()),
                               "delta_sd": np.nan, "delta_se": float(np.sqrt((sub.delta_se ** 2).sum()) / len(sub)), "cohorts": " ".join(sub.cohort)})
    return pd.concat([cells, pd.DataFrame(pooled)], ignore_index=True)


def check_against_submitted(table: pd.DataFrame, src: Path) -> int:
    with zipfile.ZipFile(src) as zf:
        hit = [n for n in zf.namelist() if n.endswith("rq2_cells.csv")]
        if len(hit) != 1:
            raise SystemExit(f"{src}: cần đúng một rq2_cells.csv, thấy {len(hit)}")
        ref = pd.read_csv(io.BytesIO(zf.read(hit[0])), dtype={"n_t": str}, encoding="utf-8-sig")
    cells = table[table.level == "cohort"]
    m = cells.merge(ref, on=["cohort", "comparison", "n_t"], suffixes=("", "_ref"))
    bad = len(m) != len(ref) or not (np.allclose(m.delta, m.delta_ref, atol=1e-12) and np.allclose(m.se, m.se_ref, atol=1e-12))
    if bad:
        raise SystemExit(f"bảng từng cohort lệch rq2_cells.csv đã nộp ({len(m)}/{len(ref)} ô ghép được)")
    return len(m)


# ------------------------------------------------------------------------------------------------- CLI
def main(argv=None, script=__file__) -> int:
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t0 = time.time()
    data, report = m2.load_inputs(args.rq2, args.cohorts)
    n_gate = m2.check_gate(report)
    shuf, _, shuf_cfg = m4.load_variant(args.rq4, "RQ2_mvp_shuffled", args.cohorts)
    if not all(c.get("source_labels_shuffled") for c in shuf_cfg.values()):
        raise SystemExit("RQ2_mvp_shuffled không ghi source_labels_shuffled = true")
    opened = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"mở kết quả lúc {opened}", flush=True)
    cohort_tab = cohort_table(data, args.n_boot)
    n_match = check_against_submitted(cohort_tab, args.rq2_tables) if args.rq2_tables else None
    dec = decomposition(data, args.n_boot)
    tvs = true_vs_shuffled(data, shuf, args.n_boot)
    sec = secondary_metrics(data)
    byk = cohort_by_k(data, args.n_boot, "gse42127") if "gse42127" in data else pd.DataFrame()
    out = results_dir(args.out)
    tables = {"qd97a_decomposition_cells": dec["cells"], "qd97a_decomposition_pooled": dec["pooled"],
              "qd97b_true_vs_shuffled_cells": tvs["cells"], "qd97b_true_vs_shuffled_pooled": tvs["pooled"],
              "qd97d_cohort_table": cohort_tab, "qd97d_secondary_metrics": sec, "qd97d_gse42127_by_k": byk}
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    files = [p for p in (args.rq2, args.rq4, args.rq2_tables) if p is not None and p.suffix == ".zip"]
    cfg = {"experiment": "RQ2_qd97_tables", "rq2_input": str(args.rq2), "rq4_input": str(args.rq4),
           "rq2_tables_input": str(args.rq2_tables) if args.rq2_tables else None, "input_sha256": {p.name: sha256(p) for p in files},
           "cohorts": args.cohorts, "k_avg": list(K_AVG), "mvp_rule": m2.MVP_RULE, "n_boot": args.n_boot, "equality_rows_ok": n_gate,
           "cohort_cells_match_submitted": n_match, "family": DESCRIPTIVE,
           "secondary_metrics_note": "chỉ số phụ: SE theo lần chạy (một tầng), không bootstrap bệnh nhân; log-loss, Brier: Δ âm = MVP tốt hơn",
           "results_opened_at_utc": opened, "elapsed_s": time.time() - t0, "versions": all_versions()}
    write_config(out, prov, cfg, data_files=files)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        p = dec["pooled"]
        print(p[p.pool_set == "gpl570"].pivot(index="comparison", columns="n_t", values="delta_ivw").round(4).to_string())
    print("ghi", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
