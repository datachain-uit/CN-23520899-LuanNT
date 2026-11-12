"""Ba phân tích mô tả QĐ126 (Review 11/11, Mục 6.3; đặc tả và quy tắc đọc chốt 11/11 trước khi tính) từ đầu ra JOB rq2_mvp và
rq4_h4b_noihc; không chạy lại mô hình. Mọi bảng là MÔ TẢ: không Holm, không đổi kết luận H2, H2c, H4a, H4b đã đăng ký, không đổi cấu hình
chính m5_pos (QĐ49). Bootstrap hai tầng của scripts/make_rq2_tables.py, khoá seed riêng "qd126", 2.000 vòng; báo từng cohort trước, rồi
gộp IVW và trọng số bằng nhau cho ba cohort GPL570 (GSE42127 báo riêng, "all4" là độ nhạy). Trung bình k ∈ {2, 3, 5}; n_T ∈ {5, 10, 20}.

(a) Độ trùng panel. Mỗi lần chạy (split, subset, n_T, k): o = |P_MVP ∩ P_SoS| / k, P_MVP = panel của m5_pos (dòng fixed đúng γ, không gian
    gen mà m5_pos ghi), P_SoS = panel của select-on-source; tương tự với pooled. Trung bình, phân vị 10 và 90 theo cohort và n_T (gộp các
    lần chạy × k ∈ {2, 3, 5}); thêm theo từng k và theo γ m5_pos chọn. Quy tắc đọc (với select-on-source; pooled báo cùng cách): trung bình
    ≥ 0,8 ở cả ba cohort GPL570 → "MVP largely reproduces the naive-transfer panel"; ≤ 0,5 ở cả ba → "MVP selects different panels with
    similar AUROC"; còn lại → "partial overlap".
(b) Khả năng lùi về target-only. Từ RQ2_mvp_shuffled: Δ(m5_cv xáo nhãn − TO) và Δ(m5_pos xáo nhãn − TO), TO lấy từ lần chạy chính cùng
    (split, subset, k); tỉ lệ γ = 0 mà m5_cv chọn theo n_T (và theo k; kèm tỉ lệ ở lần chạy chính để đối chiếu). Quy tắc đọc (M3 của H4b):
    "m5_cv falls back to target-only" khi cận dưới một phía 95 % của Δ(m5_cv xáo nhãn − TO) > −0,01 ở cả hai cách gộp; ngược lại "no
    fallback shown". Không thay kết luận H4b của m5_pos. Trước khi tính: kiểm RQ2_mvp_shuffled có đủ dòng m5_cv, cùng tập ô với m5_pos của
    RQ2_mvp.
(c) Phân rã độ ổn định (từ Φ từng nhánh của W2, scripts/make_rq4_tables.py): ΔΦ_space = Φ(TO_nguồn) − Φ(TO), ΔΦ_π = Φ(MVP) − Φ(TO_nguồn),
    ΔΦ_total = Φ(MVP) − Φ(TO) (= hiệu của H4a); SE bảo thủ √(SE²_a + SE²_b) trung bình theo k như H4a, CI 95 % hai phía; độ nhạy: CI phân vị
    của bootstrap theo split, cùng khoá "w2split" và cùng vòng lấy mẫu với rq4_h4a_cells.csv. Φ(pooled), Φ(select-on-source) = 1 đặt cạnh.

LỰA CHỌN CÀI ĐẶT (đặc tả chưa nêu, ghi vào config.json để GVHD xác nhận):
  - (a) và (b) áp quy tắc ở từng n_T ∈ {5, 10, 20} trên các cohort GPL570 có mức đó (GSE10245 không có n_T = 20); nhãn chung của một phân
    tích chỉ là "largely reproduces" / "different panels" / "falls back" khi mọi mức n_T cùng nhãn đó, ngược lại "partial overlap" /
    "no fallback shown".
  - Kịch bản của Bảng 5 gán máy móc từ hai nhãn chung: B nếu (b) "falls back"; A nếu (a) "largely reproduces" và (b) "no fallback shown";
    C nếu (a) "different panels"; ngoài ra "không khớp kịch bản nào". Đây chỉ là đọc quy tắc, đoạn đóng góp viết sau 22/11.

    python scripts/make_qd126_tables.py --rq2 rq2_mvp_1b7c021_results.zip --rq4 rq4_h4b_noihc_d74b8c0_results.zip

Đầu ra: results/survey_v2/RQ2_qd126_tables/{qd126a_overlap_runs.csv, qd126a_overlap_summary.csv, qd126a_overlap_by_k.csv,
qd126a_overlap_by_gamma.csv, qd126a_labels.csv, qd126b_fallback_cells.csv, qd126b_fallback_pooled.csv, qd126b_gamma0_share.csv,
qd126c_stability_cells.csv, qd126c_stability_pooled.csv, qd126_reading.csv, config.json}.
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


m97 = _load("make_qd97_tables_for_qd126", "make_qd97_tables.py")
m2, m4 = m97.m2, m97.m4
K_AVG = m2.K_AVG
Z975 = m2.Z975
TAG = "qd126"
N_T_MAIN = ("5", "10", "20")
D3_N_T = tuple(m2.D3_N_T)
GPL570 = tuple(m2.POOL_SETS["gpl570"])
OVERLAP_HIGH, OVERLAP_LOW = 0.8, 0.5
MARGIN = m4.MARGIN_H4B                                        # 0,01, như M3 của H4b
DESCRIPTIVE = "mô tả (QĐ126; không kiểm định)"
SHUFFLED = "mvp_shuffled"
D3_ARMS = ("select_on_source", "pooled")
LABEL_A = {"high": "MVP largely reproduces the naive-transfer panel", "low": "MVP selects different panels with similar AUROC",
           "mid": "partial overlap"}
LABEL_B = {True: "m5_cv falls back to target-only", False: "no fallback shown"}
FALLBACK = [("m5_cv_shuffled_vs_target_only", f"{SHUFFLED}|m5_cv||", "target_only", None),
            ("m5_pos_shuffled_vs_target_only", f"{SHUFFLED}|m5_pos||", "target_only", None)]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq2", type=Path, required=True, help="zip của JOB rq2_mvp (hoặc thư mục RQ2_mvp)")
    ap.add_argument("--rq4", type=Path, required=True, help="zip của JOB rq4_h4b_noihc (dùng RQ2_mvp_shuffled)")
    ap.add_argument("--cohorts", nargs="+", default=list(C.FEWSHOT_COHORTS))
    ap.add_argument("--n-boot", type=int, default=C.BOOTSTRAP_N)
    ap.add_argument("--out", default="RQ2_qd126_tables")
    return ap.parse_args(argv)


def _levels(per: pd.DataFrame, allowed=None) -> list:
    return m4._levels(per, allowed)


# ------------------------------------------------------------------------------------------------- (a) độ trùng panel
def overlap(a, b, k: int) -> float:
    return len(set(a) & set(b)) / k


def overlap_runs(data: dict, genes: dict) -> pd.DataFrame:
    """Một dòng mỗi (cohort, n_T, k, split, subset, nhánh D3): o và γ mà m5_pos chọn."""
    rows = []
    for cohort, (per, _) in data.items():
        per = per.copy()
        per["n_t"] = per.n_t.astype(str)
        for n_t in _levels(per, D3_N_T):
            for k in K_AVG:
                mvp = {(s, b): p for s, b, p in m4.panels_by_run(per, genes[cohort], n_t, k, "mvp")}
                pos = per[(per.n_t == n_t) & (per.k == k) & (per.method == "mvp") & (per.gamma_rule == m2.MVP_RULE)]
                gamma = {(int(r.split), int(r.subset)): float(r.gamma) for r in pos.itertuples()}
                for other in D3_ARMS:
                    oth = {(s, b): p for s, b, p in m4.panels_by_run(per, genes[cohort], n_t, k, other)}
                    if set(oth) != set(mvp):
                        raise ValueError(f"{cohort} n_T = {n_t}, k = {k}: {other} và MVP không cùng tập lần chạy")
                    for key in sorted(mvp):
                        rows.append({"cohort": cohort, "n_t": n_t, "k": k, "split": key[0], "subset": key[1], "other": other,
                                     "gamma_m5_pos": gamma[key], "overlap": overlap(mvp[key], oth[key], k)})
    return pd.DataFrame(rows)


def _summ(g: pd.DataFrame) -> pd.Series:
    o = g.overlap.to_numpy(float)
    return pd.Series({"n": int(len(o)), "mean": float(o.mean()), "p10": float(np.quantile(o, 0.1)), "p90": float(np.quantile(o, 0.9)),
                      "share_identical": float((o == 1).mean()), "share_disjoint": float((o == 0).mean())})


def overlap_tables(runs: pd.DataFrame) -> dict:
    keys = ["cohort", "n_t", "other"]
    summary = runs.groupby(keys, sort=False).apply(_summ, include_groups=False).reset_index()
    summary.insert(3, "k", " ".join(map(str, K_AVG)))
    by_k = runs.groupby(keys + ["k"], sort=False).apply(_summ, include_groups=False).reset_index()
    by_gamma = runs.groupby(keys + ["gamma_m5_pos"], sort=False).apply(_summ, include_groups=False).reset_index()
    return {"summary": summary, "by_k": by_k, "by_gamma": by_gamma, "labels": overlap_labels(summary)}


def _label_a(means) -> str:
    means = np.asarray(means, float)
    if len(means) and (means >= OVERLAP_HIGH).all():
        return LABEL_A["high"]
    if len(means) and (means <= OVERLAP_LOW).all():
        return LABEL_A["low"]
    return LABEL_A["mid"]


def overlap_labels(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for other in D3_ARMS:
        s = summary[(summary.other == other) & summary.cohort.isin(GPL570)]
        per_level = []
        for n_t in [v for v in N_T_MAIN if v in set(s.n_t)]:
            g = s[s.n_t == n_t]
            lab = _label_a(g["mean"])
            per_level.append(lab)
            rows.append({"analysis": "a", "other": other, "n_t": n_t, "cohorts": " ".join(g.cohort), "means": " ".join(f"{v:.4f}" for v in g["mean"]),
                         "label": lab})
        overall = per_level[0] if per_level and len(set(per_level)) == 1 else LABEL_A["mid"]
        rows.append({"analysis": "a", "other": other, "n_t": "all", "cohorts": " ".join(GPL570), "means": "", "label": overall})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- (b) lùi về target-only
def _cells_of(per: pd.DataFrame, rule: str, method: str = "mvp") -> set:
    p = per[(per.method == method) & (per.gamma_rule == rule)]
    return set(zip(p.n_t.astype(str), p.k.astype(int), p.split.astype(int), p.subset.astype(int)))


def check_shuffled(data: dict, shuffled: dict) -> dict:
    """Trước khi tính (b): RQ2_mvp_shuffled phải có dòng m5_cv ở đúng mọi ô m5_pos của RQ2_mvp, không trùng; trả số dòng m5_cv mỗi cohort."""
    counts = {}
    for c, (per, _) in data.items():
        sp = shuffled[c][0]
        cv = sp[(sp.method == "mvp") & (sp.gamma_rule == "m5_cv")]
        if cv.duplicated(["n_t", "k", "split", "subset"]).any():
            raise SystemExit(f"{c}: RQ2_mvp_shuffled có dòng m5_cv trùng (n_T, k, split, subset)")
        if cv.auroc.isna().any():
            raise SystemExit(f"{c}: RQ2_mvp_shuffled có dòng m5_cv thiếu AUROC")
        if _cells_of(sp, "m5_cv") != _cells_of(per, m2.MVP_RULE):
            raise SystemExit(f"{c}: tập ô m5_cv của RQ2_mvp_shuffled khác tập ô m5_pos của RQ2_mvp (QĐ126b: báo GVHD trước khi tính)")
        counts[c] = int(len(cv))
    return counts


def merge_shuffled(data: dict, shuffled: dict) -> dict:
    """Lần chạy chính + các dòng MVP của lần xáo nhãn (đổi method thành mvp_shuffled); target-only luôn lấy từ lần chạy chính."""
    merged = {}
    for c, (per, sc) in data.items():
        sp, ss = m4.relabel(*shuffled[c], SHUFFLED)
        sp, ss = sp[sp.method == SHUFFLED], ss[ss.method == SHUFFLED]
        merged[c] = (pd.concat([per, sp], ignore_index=True), pd.concat([sc, ss], ignore_index=True))
    return merged


def fallback_tables(data: dict, shuffled: dict, n_boot: int) -> dict:
    out = m97.paired(merge_shuffled(data, shuffled), FALLBACK, n_boot, tag=TAG)
    pooled = out["pooled"]
    # IVW không xác định khi một cohort có SE = 0 (m5_cv chọn γ = 0 ở mọi lần chạy nên trùng TO, Δ = 0 đúng): khi đó chỉ xét bản gộp
    # trọng số bằng nhau và đánh dấu ivw_undefined (lựa chọn cài đặt, chốt trước khi mở kết quả)
    pooled["ivw_undefined"] = pooled.delta_ivw.isna()
    pooled["noninferior"] = ((pooled.lower_ivw > -MARGIN) | pooled.ivw_undefined) & (pooled.lower_eq > -MARGIN)
    pooled["label"] = np.where(pooled.comparison == FALLBACK[0][0],
                               pooled.noninferior.map(LABEL_B) + np.where(pooled.ivw_undefined, " (IVW không xác định: chỉ trọng số bằng nhau)", ""), "")
    pooled["family"] = DESCRIPTIVE
    return {"cells": out["cells"], "pooled": pooled}


def fallback_labels(pooled: pd.DataFrame) -> pd.DataFrame:
    p = pooled[(pooled.comparison == FALLBACK[0][0]) & (pooled.pool_set == "gpl570") & pooled.n_t.isin(N_T_MAIN)]
    rows = [{"analysis": "b", "other": "", "n_t": r.n_t, "cohorts": r.cohorts, "means": f"lower_ivw={r.lower_ivw:.4f} lower_eq={r.lower_eq:.4f}",
             "label": LABEL_B[bool(r.noninferior)], "note": "IVW không xác định (một cohort SE = 0): chỉ trọng số bằng nhau"
             if getattr(r, "ivw_undefined", False) else ""} for r in p.itertuples()]
    ok = bool(len(p)) and bool(p.noninferior.all())
    rows.append({"analysis": "b", "other": "", "n_t": "all", "cohorts": " ".join(GPL570), "means": "", "label": LABEL_B[ok]})
    return pd.DataFrame(rows)


def gamma0_share(data: dict, shuffled: dict) -> pd.DataFrame:
    rows = []
    for source, frames in (("shuffled", shuffled), ("true", data)):
        for c, (per, _) in frames.items():
            p = per[(per.method == "mvp") & (per.gamma_rule == "m5_cv") & per.k.isin(K_AVG)].copy()
            p["n_t"] = p.n_t.astype(str)
            for n_t, g in p.groupby("n_t", sort=False):
                rows.append({"source_labels": source, "cohort": c, "n_t": n_t, "k": " ".join(map(str, K_AVG)), "n": int(len(g)),
                             "share_gamma0": float((g.gamma == 0).mean())})
                for k, gk in g.groupby("k"):
                    rows.append({"source_labels": source, "cohort": c, "n_t": n_t, "k": str(int(k)), "n": int(len(gk)),
                                 "share_gamma0": float((gk.gamma == 0).mean())})
    out = pd.DataFrame(rows)
    return out.sort_values(["source_labels", "cohort", "k", "n_t"], key=lambda s: s.map(m2._n_t_key) if s.name == "n_t" else s,
                           kind="stable", ignore_index=True)


# ------------------------------------------------------------------------------------------------- (c) phân rã độ ổn định
COMPONENTS = [("dphi_total", "mvp", "target_only"), ("dphi_space", m4.TO_SOURCE, "target_only"), ("dphi_pi", "mvp", m4.TO_SOURCE)]


def stability_cells(data: dict, genes: dict, space: dict, n_boot: int) -> pd.DataFrame:
    rows = []
    for cohort, (per, _) in data.items():
        for n_t in _levels(per):
            runs = m4.arm_panels(per, genes[cohort], n_t)
            for need in ("mvp", "target_only", m4.TO_SOURCE):
                if need not in runs:
                    raise ValueError(f"{cohort} n_T = {n_t}: thiếu panel {need}")
            stats = {arm: [m4._phi([p for _, _, p in runs[arm][k]], space[cohort]) for k in K_AVG] for arm in runs}
            phi = {arm: float(np.mean([x["stability"] for x in st])) for arm, st in stats.items()}
            seed = m2.cell_seed(m4.SPLIT_BOOT_TAG, cohort, n_t, *K_AVG)
            bs = m4.split_bootstrap({a: r for a, r in runs.items() if a != m4.SOS}, space[cohort], n_boot, seed) if n_boot else {}
            for name, a, b in COMPONENTS:
                d = phi[a] - phi[b]
                se = float(np.mean([np.sqrt(x["se"] ** 2 + y["se"] ** 2) for x, y in zip(stats[a], stats[b])]))
                row = {"cohort": cohort, "n_t": n_t, "comparison": name, "arm_a": m4.label(a), "arm_b": m4.label(b), "k": " ".join(map(str, K_AVG)),
                       "phi_a": phi[a], "phi_b": phi[b], "delta": d, "se": se, "ci95_low": d - Z975 * se, "ci95_high": d + Z975 * se,
                       "phi_pooled": phi.get("pooled", np.nan), "phi_select_on_source": phi.get(m4.SOS, np.nan), "d": int(space[cohort])}
                if bs:
                    diff = bs[a] - bs[b]
                    row.update({"boot_low": m4._q(diff, 0.025), "boot_high": m4._q(diff, 0.975), "boot_n": int(n_boot), "boot_seed": int(seed)})
                rows.append(row)
    return pd.DataFrame(rows)


def stability_pooled(cells: pd.DataFrame) -> pd.DataFrame:
    out = []
    sets = {**m2.POOL_SETS, "all4": tuple(cells.cohort.unique())}
    for (name, n_t), g in cells.groupby(["comparison", "n_t"], sort=False):
        for set_name, members in sets.items():
            sub = g[g.cohort.isin(members)]
            if len(sub):
                out.append({"comparison": name, "n_t": n_t, "pool_set": set_name, "family": DESCRIPTIVE, "cohorts": " ".join(sub.cohort),
                            **m2.pool(sub[["cohort", "delta", "se"]], 0.0)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------- đọc theo Bảng 5
def scenario(label_a: str, label_b: str) -> str:
    if label_b == LABEL_B[True]:
        return "B"
    if label_a == LABEL_A["high"] and label_b == LABEL_B[False]:
        return "A"
    if label_a == LABEL_A["low"]:
        return "C"
    return "không khớp kịch bản nào"


def reading(labels_a: pd.DataFrame, labels_b: pd.DataFrame) -> pd.DataFrame:
    la = labels_a[(labels_a.other == "select_on_source") & (labels_a.n_t == "all")].label.item()
    lb = labels_b[labels_b.n_t == "all"].label.item()
    rows = pd.concat([labels_a, labels_b], ignore_index=True)
    return pd.concat([rows, pd.DataFrame([{"analysis": "bang5", "other": "", "n_t": "all", "cohorts": " ".join(GPL570), "means": f"(a) {la}; (b) {lb}",
                                           "label": f"kịch bản {scenario(la, lb)}"}])], ignore_index=True)


# ------------------------------------------------------------------------------------------------- CLI
def main(argv=None, script=__file__) -> int:
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t0 = time.time()
    data, report = m2.load_inputs(args.rq2, args.cohorts)
    n_gate = m2.check_gate(report)
    _, genes, main_cfg = m4.load_variant(args.rq2, "RQ2_mvp", args.cohorts)
    shuf, _, shuf_cfg = m4.load_variant(args.rq4, "RQ2_mvp_shuffled", args.cohorts)
    if not all(c.get("source_labels_shuffled") for c in shuf_cfg.values()):
        raise SystemExit("RQ2_mvp_shuffled không ghi source_labels_shuffled = true")
    n_cv = check_shuffled(data, shuf)
    space = {c: int(main_cfg[c]["gene_space_size"]) for c in args.cohorts}
    opened = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"mở kết quả QĐ126 lúc {opened}", flush=True)
    ov_runs = overlap_runs(data, genes)
    ov = overlap_tables(ov_runs)
    fb = fallback_tables(data, shuf, args.n_boot)
    lab_b = fallback_labels(fb["pooled"])
    g0 = gamma0_share(data, shuf)
    st_c = stability_cells(data, genes, space, args.n_boot)
    st_p = stability_pooled(st_c)
    read = reading(ov["labels"], lab_b)
    out = results_dir(args.out)
    tables = {"qd126a_overlap_runs": ov_runs, "qd126a_overlap_summary": ov["summary"], "qd126a_overlap_by_k": ov["by_k"],
              "qd126a_overlap_by_gamma": ov["by_gamma"], "qd126a_labels": ov["labels"], "qd126b_fallback_cells": fb["cells"],
              "qd126b_fallback_pooled": fb["pooled"], "qd126b_gamma0_share": g0, "qd126c_stability_cells": st_c, "qd126c_stability_pooled": st_p,
              "qd126_reading": read}
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    files = [p for p in (args.rq2, args.rq4) if p.suffix == ".zip"]
    cfg = {"experiment": "RQ2_qd126_tables", "rq2_input": str(args.rq2), "rq4_input": str(args.rq4), "input_sha256": {p.name: sha256(p) for p in files},
           "cohorts": args.cohorts, "k_avg": list(K_AVG), "n_t_main": list(N_T_MAIN), "mvp_rule": m2.MVP_RULE, "n_boot": args.n_boot, "seed_key": TAG,
           "stability_bootstrap_seed_key": m4.SPLIT_BOOT_TAG, "equality_rows_ok": n_gate, "shuffled_m5_cv_rows": n_cv, "gene_space_size": space,
           "family": DESCRIPTIVE, "overlap_thresholds": [OVERLAP_HIGH, OVERLAP_LOW], "fallback_margin": MARGIN,
           "choices_to_confirm": [
               "(a), (b): quy tắc áp ở từng n_T ∈ {5, 10, 20} trên các cohort GPL570 có mức đó; nhãn chung chỉ khi mọi mức cùng nhãn",
               "(a): mean, p10, p90 gộp các lần chạy × k ∈ {2, 3, 5}; quy tắc đọc theo select-on-source, pooled báo cùng cách",
               "(c): SE bảo thủ như H4a; CI phân vị bootstrap theo split dùng chung vòng lấy mẫu với rq4_h4a_cells.csv",
               "kịch bản Bảng 5 gán máy móc từ hai nhãn chung (qd126_reading.csv)",
               "(b): khi IVW không xác định vì một cohort có SE = 0 (m5_cv trùng TO ở mọi lần chạy, Δ = 0 đúng), quy tắc chỉ xét bản gộp "
               "trọng số bằng nhau; cột ivw_undefined đánh dấu"],
           "results_opened_at_utc": opened, "elapsed_s": time.time() - t0, "versions": all_versions()}
    write_config(out, prov, cfg, data_files=files)
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.max_colwidth", 80):
        print(read.to_string(index=False))
    print("ghi", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
