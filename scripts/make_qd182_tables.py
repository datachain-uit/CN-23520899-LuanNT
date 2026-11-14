"""QĐ182 (Review 12/11 §5; QĐ190): đối chứng âm thăm dò — chuyển giao đơn giản với nhãn nguồn xáo. Đặc tả và quy tắc đọc chốt 12/11,
script viết, commit và push TRƯỚC khi mở kết quả RQ2_d3_shuffled (thứ tự mù QĐ69b). Không chạy lại mô hình.

Bốn hiệu (Δ = a − b), trung bình k ∈ {2, 3, 5}, n_T ∈ {5, 10, 20}, bootstrap hai tầng của scripts/make_rq2_tables.py, khoá seed "qd182",
2.000 vòng; từng cohort trước, rồi gộp IVW và trọng số bằng nhau cho ba cohort GPL570; GSE42127 báo riêng ("all4" là độ nhạy):
  1. sos_shuffled_vs_target_only        : select-on-source (nguồn xáo nhãn) − target-only (lần chạy chính)
  2. pooled_shuffled_vs_target_only     : pooled (nguồn xáo nhãn) − target-only
  3. m5_cv_shuffled_vs_sos_shuffled     : m5_cv (nguồn xáo, RQ2_mvp_shuffled) − select-on-source (nguồn xáo)
  4. m5_cv_shuffled_vs_pooled_shuffled  : m5_cv (nguồn xáo) − pooled (nguồn xáo)
Hoán vị nhãn nguồn của RQ2_d3_shuffled phải trùng RQ2_mvp_shuffled (cùng shuffle_seed, cùng shuffle_label_agreement, cùng SHA-256
nguồn): kiểm trước khi đọc bất kỳ AUROC nào; lệch thì dừng.

Quy tắc đọc (mô tả, không Holm, không đổi H4b và cấu hình chính m5_pos), ở từng n_T trên bản gộp GPL570:
  * "naive transfer is harmed by an uninformative source" (cho select-on-source, pooled riêng) khi cận trên một phía 95 % của
    Δ(chiến lược xáo − TO) < −0,01 ở CẢ HAI cách gộp; ngược lại "harm not shown".
  * "m5_cv loses less than naive transfer" (so với từng chiến lược) khi cận dưới một phía 95 % của Δ(m5_cv xáo − chiến lược xáo) > 0
    ở CẢ HAI cách gộp; ngược lại "not shown".

LỰA CHỌN CÀI ĐẶT (đặc tả chưa nêu; ghi vào config.json, chốt trước khi mở):
  - IVW không xác định khi một cohort có SE = 0 (hai nhánh trùng hệt ở mọi lần chạy): quy tắc chỉ xét bản trọng số bằng nhau, cột
    ivw_undefined đánh dấu (như QĐ126b).
  - Nhãn chung (n_T = "all") chỉ khi mọi mức n_T cùng nhãn dương; ngược lại nhãn âm (như QĐ126).
  - n_T = 20 chỉ có GSE30219, GSE37745 trong bản gộp GPL570 (GSE10245 không có mức này).

    python scripts/make_qd182_tables.py --rq2 rq2_mvp_1b7c021_results.zip --rq4 rq4_h4b_noihc_d74b8c0_results.zip \\
        --d3 qd182_d3_shuffled_c36c8c7_results.zip

Đầu ra: results/survey_v2/RQ2_qd182_tables/{qd182_cells.csv, qd182_pooled.csv, qd182_gse42127.csv, qd182_labels.csv, config.json}.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
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


m97 = _load("make_qd97_tables_for_qd182", "make_qd97_tables.py")
m2, m4 = m97.m2, m97.m4
TAG = "qd182"
N_T_MAIN = ("5", "10", "20")
GPL570 = tuple(m2.POOL_SETS["gpl570"])
MARGIN = 0.01
Z95 = m2.Z95
DESCRIPTIVE = "mô tả (QĐ182; thăm dò, không Holm)"
SHUF_MVP, SHUF_SOS, SHUF_POOLED = "mvp_shuffled", "sos_shuffled", "pooled_shuffled"
CV = f"{SHUF_MVP}|m5_cv||"
SPECS = [("sos_shuffled_vs_target_only", SHUF_SOS, "target_only", N_T_MAIN),
         ("pooled_shuffled_vs_target_only", SHUF_POOLED, "target_only", N_T_MAIN),
         ("m5_cv_shuffled_vs_sos_shuffled", CV, SHUF_SOS, N_T_MAIN),
         ("m5_cv_shuffled_vs_pooled_shuffled", CV, SHUF_POOLED, N_T_MAIN)]
HARM = {"sos_shuffled_vs_target_only": "select_on_source", "pooled_shuffled_vs_target_only": "pooled"}
LESS = {"m5_cv_shuffled_vs_sos_shuffled": "select_on_source", "m5_cv_shuffled_vs_pooled_shuffled": "pooled"}
LABEL_HARM = {True: "naive transfer is harmed by an uninformative source", False: "harm not shown"}
LABEL_LESS = {True: "m5_cv loses less than naive transfer", False: "not shown"}
PERM_KEYS = ("shuffle_seed", "shuffle_label_agreement", "shuffle_rule")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq2", type=Path, required=True, help="zip của JOB rq2_mvp (target-only của lần chạy chính)")
    ap.add_argument("--rq4", type=Path, required=True, help="zip của JOB rq4_h4b_noihc (RQ2_mvp_shuffled: m5_cv nguồn xáo)")
    ap.add_argument("--d3", type=Path, required=True, help="zip của JOB qd182_d3_shuffled (RQ2_d3_shuffled)")
    ap.add_argument("--cohorts", nargs="+", default=list(C.FEWSHOT_COHORTS))
    ap.add_argument("--n-boot", type=int, default=C.BOOTSTRAP_N)
    ap.add_argument("--out", default="RQ2_qd182_tables")
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------------------------- kiểm trước khi mở
def check_permutation(mvp_cfg: dict, d3_cfg: dict) -> dict:
    """Hoán vị nhãn nguồn của RQ2_d3_shuffled = của RQ2_mvp_shuffled, từng cohort (chỉ đọc config, không đọc AUROC)."""
    out = {}
    for c in d3_cfg:
        a, b = mvp_cfg[c], d3_cfg[c]
        if not (a.get("source_labels_shuffled") and b.get("source_labels_shuffled")):
            raise SystemExit(f"{c}: thiếu source_labels_shuffled = true")
        diff = [k for k in PERM_KEYS if a.get(k) != b.get(k)]
        if a.get("source_sha256") != b.get("source_sha256"):
            diff.append("source_sha256")
        if diff:
            raise SystemExit(f"{c}: hoán vị nhãn nguồn của RQ2_d3_shuffled khác RQ2_mvp_shuffled ở {diff} — dừng, báo GVHD")
        out[c] = {k: b.get(k) for k in PERM_KEYS}
    return out


def _rename(per: pd.DataFrame, sc: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    per, sc = per[per.method.isin(mapping)].copy(), sc.copy()
    if "method" not in sc:
        raise ValueError("scores của RQ2_d3_shuffled thiếu cột method")
    sc = sc[sc.method.isin(mapping)].copy()
    per["method"] = per.method.map(mapping)
    sc["method"] = sc.method.map(mapping)
    return per, sc


def check_cells(main: dict, d3: dict, shuffled: dict) -> dict:
    """Nhánh D3 xáo và m5_cv xáo phải có đúng mọi ô (n_T ∈ {5, 10, 20}, k, split, subset) của target-only lần chính, không trùng."""
    counts = {}
    for c, (per, _) in main.items():
        to = per[(per.method == "mvp") & (per.gamma_rule == "fixed") & (per.gamma == 0) & (per.prefilter_space == "target")
                 & per.n_t.astype(str).isin(N_T_MAIN)]
        want = set(zip(to.n_t.astype(str), to.k.astype(int), to.split.astype(int), to.subset.astype(int)))
        for arm, frame in (("select_on_source", d3[c][0]), ("pooled", d3[c][0])):
            f = frame[frame.method == arm]
            got = list(zip(f.n_t.astype(str), f.k.astype(int), f.split.astype(int), f.subset.astype(int)))
            if len(got) != len(set(got)) or set(got) != want:
                raise SystemExit(f"{c}: {arm} của RQ2_d3_shuffled không đúng tập ô của target-only lần chính")
        sp = shuffled[c][0]
        cv = sp[(sp.method == "mvp") & (sp.gamma_rule == "m5_cv") & sp.n_t.astype(str).isin(N_T_MAIN)]
        got = set(zip(cv.n_t.astype(str), cv.k.astype(int), cv.split.astype(int), cv.subset.astype(int)))
        if got != want:
            raise SystemExit(f"{c}: m5_cv của RQ2_mvp_shuffled không đúng tập ô của target-only lần chính")
        counts[c] = int(len(want))
    return counts


def merge(main: dict, shuffled: dict, d3: dict) -> dict:
    out = {}
    for c, (per, sc) in main.items():
        sp, ss = m4.relabel(*shuffled[c], SHUF_MVP)
        sp, ss = sp[sp.method == SHUF_MVP], ss[ss.method == SHUF_MVP]
        dp, ds = _rename(*d3[c], {"select_on_source": SHUF_SOS, "pooled": SHUF_POOLED})
        out[c] = (pd.concat([per, sp, dp], ignore_index=True), pd.concat([sc, ss, ds], ignore_index=True))
    return out


# ------------------------------------------------------------------------------------------------- bảng và nhãn
def tables(merged: dict, n_boot: int) -> dict:
    out = m97.paired(merged, SPECS, n_boot, tag=TAG)
    pooled = out["pooled"].copy()
    pooled["family"] = DESCRIPTIVE
    pooled["ivw_undefined"] = pooled.delta_ivw.isna()
    pooled["upper_ivw"] = pooled.delta_ivw + Z95 * pooled.se_ivw
    pooled["upper_eq"] = pooled.delta_eq + Z95 * pooled.se_eq
    harm = ((pooled.upper_ivw < -MARGIN) | pooled.ivw_undefined) & (pooled.upper_eq < -MARGIN)
    less = ((pooled.lower_ivw > 0) | pooled.ivw_undefined) & (pooled.lower_eq > 0)
    pooled["rule_met"] = np.where(pooled.comparison.isin(list(HARM)), harm, less)
    return {"cells": out["cells"], "pooled": pooled}


def labels(pooled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comp in [s[0] for s in SPECS]:
        p = pooled[(pooled.comparison == comp) & (pooled.pool_set == "gpl570") & pooled.n_t.isin(N_T_MAIN)]
        names = LABEL_HARM if comp in HARM else LABEL_LESS
        strategy = HARM.get(comp) or LESS.get(comp)
        for r in p.itertuples():
            bound = (f"upper_ivw={r.upper_ivw:.4f} upper_eq={r.upper_eq:.4f}" if comp in HARM
                     else f"lower_ivw={r.lower_ivw:.4f} lower_eq={r.lower_eq:.4f}")
            rows.append({"comparison": comp, "strategy": strategy, "n_t": r.n_t, "cohorts": r.cohorts, "bounds": bound,
                         "label": names[bool(r.rule_met)], "note": "IVW không xác định (một cohort SE = 0): chỉ trọng số bằng nhau"
                         if r.ivw_undefined else ""})
        ok = bool(len(p)) and bool(p.rule_met.all())
        rows.append({"comparison": comp, "strategy": strategy, "n_t": "all", "cohorts": " ".join(GPL570), "bounds": "", "label": names[ok], "note": ""})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- CLI
def main(argv=None, script=__file__) -> int:
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t0 = time.time()
    main_data, report = m2.load_inputs(args.rq2, args.cohorts)
    n_gate = m2.check_gate(report)
    shuf, _, shuf_cfg = m4.load_variant(args.rq4, "RQ2_mvp_shuffled", args.cohorts)
    d3, _, d3_cfg = m4.load_variant(args.d3, "RQ2_d3_shuffled", args.cohorts)
    perm = check_permutation(shuf_cfg, d3_cfg)
    n_cells = check_cells(main_data, d3, shuf)
    opened = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"mở kết quả QĐ182 lúc {opened}", flush=True)
    t = tables(merge(main_data, shuf, d3), args.n_boot)
    lab = labels(t["pooled"])
    out = results_dir(args.out)
    t["cells"].to_csv(out / "qd182_cells.csv", index=False, encoding="utf-8-sig")
    t["pooled"].to_csv(out / "qd182_pooled.csv", index=False, encoding="utf-8-sig")
    t["cells"][t["cells"].cohort == "gse42127"].to_csv(out / "qd182_gse42127.csv", index=False, encoding="utf-8-sig")
    lab.to_csv(out / "qd182_labels.csv", index=False, encoding="utf-8-sig")
    files = [p for p in (args.rq2, args.rq4, args.d3) if p.suffix == ".zip"]
    cfg = {"experiment": "RQ2_qd182_tables", "decision": "QĐ182 (Review 12/11 §5), QĐ190", "rq2_input": str(args.rq2), "rq4_input": str(args.rq4),
           "d3_input": str(args.d3), "input_sha256": {p.name: sha256(p) for p in files}, "cohorts": args.cohorts, "k_avg": list(m2.K_AVG),
           "n_t_main": list(N_T_MAIN), "n_boot": args.n_boot, "seed_key": TAG, "margin": MARGIN, "family": DESCRIPTIVE,
           "equality_rows_ok": n_gate, "permutation_check": perm, "cells_per_cohort": n_cells,
           "rules": {"harm": "one-sided 95 % upper bound of Δ(strategy shuffled − TO) < −0.01 under IVW and equal weights (GPL570)",
                     "less": "one-sided 95 % lower bound of Δ(m5_cv shuffled − strategy shuffled) > 0 under IVW and equal weights (GPL570)"},
           "choices_to_confirm": [
               "IVW undefined (a cohort with SE = 0): rule uses equal weights only, flagged in ivw_undefined (as QĐ126b)",
               "overall label (n_T = all) only when every n_T level meets the rule (as QĐ126)",
               "n_T = 20 pools GSE30219 and GSE37745 only (GSE10245 has no n_T = 20)"],
           "results_opened_at_utc": opened, "elapsed_s": time.time() - t0, "versions": all_versions()}
    write_config(out, prov, cfg, data_files=files)
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.max_colwidth", 80):
        print(lab.to_string(index=False))
    print("ghi", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
