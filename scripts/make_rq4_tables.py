"""RQ4 và ablation RQ2 (Review 07/11 Bảng 8; KBTN v2.5 Mục 4, 9) từ đầu ra của JOB rq2_mvp và rq4_h4b_noihc; không chạy lại mô hình.

Dùng chung máy bootstrap hai tầng, ghép cặp và gộp của scripts/make_rq2_tables.py (QĐ69a; MVP = m5_pos, k ∈ {2, 3, 5}).

H4b (chỉ báo cáo, M3): MVP với π từ TCGA xáo nhãn (RQ2_mvp_shuffled) so với target-only trong cùng lần chạy; biên 0,01 như H2c; gộp IVW và
  trọng số bằng nhau (GPL570; 4 cohort là độ nhạy); cận dưới một phía 95 % ≤ −0,01 thì ghi "không chứng minh được không thua kém".
H4a: độ ổn định Nogueira (survey_v2.stability.nogueira, cùng mã với E4) của panel MVP, target-only, pooled (và select-on-source, chỉ báo kèm;
  QĐ97c) qua 20 lần chạy mỗi (cohort, n_T, k), CI 95 % từng nhánh; quyết định chỉ theo MVP − target-only, MVP − pooled là mô tả;
  d = cỡ không gian gen G. LỰA CHỌN CÀI ĐẶT (KBTN chưa nêu, cần GVHD xác nhận): Δ = Φ(MVP) − Φ(TO) lấy trung bình k ∈ {2, 3, 5};
  SE của hiệu = √(SE²_MVP + SE²_TO) theo phương sai tiệm cận của Nogueira (bảo thủ: bỏ qua tương quan dương giữa hai nhánh), SE của trung
  bình theo k = trung bình các SE (cận trên); gộp IVW và trọng số bằng nhau qua GPL570; một mức n_T ≤ 20 đạt khi Δ ≥ 0,1 và CI 95 % hai
  phía không chứa 0 theo cả hai bản gộp. Ủng hộ: đạt ở cả 3 mức; một phần: ở một số mức; không ủng hộ: không mức nào.
Ablation (mô tả, không kiểm định): phân rã MVP − TO = (TO_nguồn − TO) + (MVP − TO_nguồn) với TO_nguồn = γ = 0 trên không gian gen của nguồn;
  m5_cv − m5_pos; từng γ cố định − m5_pos; loại 4 gen IHC (RQ2_mvp_noIHC − RQ2_mvp, cho MVP và target-only). Phân bố γ được chọn.

    python scripts/make_rq4_tables.py --rq2 rq2_mvp_1b7c021_results.zip --rq4 rq4_h4b_noihc_d74b8c0_results.zip

Đầu ra: results/survey_v2/RQ4_tables/{rq4_h4b_cells.csv, rq4_h4b_pooled.csv, rq4_h4a_cells.csv, rq4_h4a_pooled.csv, rq4_decisions.csv,
rq2_ablation_cells.csv, rq2_ablation_pooled.csv, rq2_noihc_cells.csv, rq2_noihc_pooled.csv, rq2_gamma_distribution.csv, config.json}.
"""

from __future__ import annotations

import argparse
import io
import importlib.util
import json
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
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402
from survey_v2.stability import nogueira  # noqa: E402


def _load_rq2():
    spec = importlib.util.spec_from_file_location("make_rq2_tables_for_rq4", REPO / "scripts" / "make_rq2_tables.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


m2 = _load_rq2()
K_AVG = m2.K_AVG
H4_N_T = ("5", "10", "20")
MARGIN_H4B = 0.01
H4A_MIN_DELTA = 0.1
M3_NOTE = "không chứng minh được không thua kém"
DESCRIPTIVE = "mô tả (ablation; không kiểm định)"
TO_SOURCE = "mvp|fixed|0|source"
ABLATION = [  # (tên, nhánh a, nhánh b): Δ = a − b
    ("decomp_total", "mvp", "target_only"),
    ("decomp_space", TO_SOURCE, "target_only"),
    ("decomp_pi", "mvp", TO_SOURCE),
    ("m5_cv_vs_m5_pos", "mvp|m5_cv||", "mvp"),
    ("gamma_0.5_vs_m5_pos", "mvp|fixed|0.5|source", "mvp"),
    ("gamma_1_vs_m5_pos", "mvp|fixed|1|source", "mvp"),
    ("gamma_2_vs_m5_pos", "mvp|fixed|2|source", "mvp"),
]
NOIHC = [("noihc_mvp", "mvp_noihc|m5_pos||", "mvp"), ("noihc_target_only", "mvp_noihc|fixed|0|target", "target_only")]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq2", type=Path, required=True, help="zip của JOB rq2_mvp (hoặc thư mục RQ2_mvp)")
    ap.add_argument("--rq4", type=Path, required=True, help="zip của JOB rq4_h4b_noihc (RQ2_mvp_shuffled, RQ2_mvp_noIHC)")
    ap.add_argument("--cohorts", nargs="+", default=list(C.FEWSHOT_COHORTS))
    ap.add_argument("--n-boot", type=int, default=C.BOOTSTRAP_N)
    ap.add_argument("--out", default="RQ4_tables")
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------------------------- đọc
def load_variant(src: Path, folder: str, cohorts) -> tuple[dict, dict, dict]:
    """({cohort: (per_run, scores)}, {cohort: selected_genes}, {cohort: config}) của thư mục folder (RQ2_mvp, RQ2_mvp_shuffled, ...)."""
    src = Path(src)
    data, genes, cfgs = {}, {}, {}
    names = ("rq2_mvp_per_run.csv", "rq2_mvp_scores.csv", "rq2_mvp_selected_genes.csv", "config.json")
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as zf:
            members = zf.namelist()
            for c in cohorts:
                got = {}
                for name in names:
                    hit = [n for n in members if n.endswith(f"{folder}/{c}/{name}")]
                    if len(hit) != 1:
                        raise ValueError(f"{src}: cần đúng một {folder}/{c}/{name}, thấy {len(hit)}")
                    raw = zf.read(hit[0])
                    got[name] = json.loads(raw) if name.endswith(".json") else m2._read(io.BytesIO(raw))
                data[c], genes[c], cfgs[c] = (got[names[0]], got[names[1]]), got[names[2]], got[names[3]]
        return data, genes, cfgs
    root = src if src.name == folder else src / folder
    for c in cohorts:
        data[c] = (m2._read(root / c / names[0]), m2._read(root / c / names[1]))
        genes[c] = m2._read(root / c / names[2])
        cfgs[c] = json.loads((root / c / names[3]).read_text(encoding="utf-8"))
    return data, genes, cfgs


def relabel(per: pd.DataFrame, sc: pd.DataFrame, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    per, sc = per.copy(), sc.copy()
    per.loc[per.method == "mvp", "method"] = method
    if "method" not in sc:
        sc["method"] = "mvp"
    sc.loc[sc.method == "mvp", "method"] = method
    return per, sc


# ------------------------------------------------------------------------------------------------- so sánh ghép cặp
def _levels(per: pd.DataFrame, allowed=None) -> list:
    lv = sorted(per.n_t.astype(str).unique(), key=m2._n_t_key)
    return lv if allowed is None else [v for v in lv if v in allowed]


def compare(data: dict, specs, n_boot: int, margin: float = 0.0, allowed=None, tag: str = "rq4") -> tuple[pd.DataFrame, pd.DataFrame]:
    """specs = [(tên, nhánh a, nhánh b)]; Δ = a − b, trung bình k ∈ K_AVG; bootstrap hai tầng; gộp GPL570 và 4 cohort."""
    cells = []
    for cohort, (per, sc) in data.items():
        for n_t in _levels(per, allowed):
            for name, a, b in specs:
                runs = m2.collect_runs(per, sc, cohort, n_t, a, b, K_AVG)
                cell = m2.bootstrap_cell(runs, n_boot, m2.cell_seed(tag, cohort, n_t, name, *K_AVG))
                cells.append({"cohort": cohort, "comparison": name, "arm_a": a, "arm_b": b, "n_t": n_t, **cell,
                              "auroc_a_mean": float(np.mean([np.mean(list(r.auroc_a.values())) for r in runs])),
                              "auroc_b_mean": float(np.mean([np.mean(list(r.auroc_b.values())) for r in runs]))})
    cells = pd.DataFrame(cells)
    pooled = []
    sets = {**m2.POOL_SETS, "all4": tuple(data)}
    for (name, n_t), g in cells.groupby(["comparison", "n_t"], sort=False):
        for set_name, members in sets.items():
            sub = g[g.cohort.isin(members)]
            if len(sub):
                pooled.append({"comparison": name, "n_t": n_t, "pool_set": set_name, "cohorts": " ".join(sub.cohort), **m2.pool(sub, margin)})
    return cells, pd.DataFrame(pooled)


def ablation_tables(data: dict, n_boot: int) -> dict:
    cells, pooled = compare(data, ABLATION, n_boot, tag="rq2abl")
    pooled.insert(3, "family", DESCRIPTIVE)
    return {"cells": cells, "pooled": pooled}


def noihc_tables(data: dict, noihc: dict, n_boot: int) -> dict:
    merged = {}
    for c, (per, sc) in data.items():
        if c in noihc:
            vp, vs = relabel(*noihc[c], "mvp_noihc")
            merged[c] = (pd.concat([per, vp], ignore_index=True), pd.concat([sc, vs], ignore_index=True))
    cells, pooled = compare(merged, NOIHC, n_boot, tag="rq2noihc")
    pooled.insert(3, "family", DESCRIPTIVE)
    return {"cells": cells, "pooled": pooled}


def h4b_tables(shuffled: dict, n_boot: int) -> dict:
    cells, pooled = compare(shuffled, [("h4b_mvp_shuffled_vs_target_only", "mvp", "target_only")], n_boot, margin=MARGIN_H4B, tag="rq4h4b")
    pooled["p_holm"] = np.nan
    pooled["noninferior"] = (pooled.lower_ivw > -MARGIN_H4B) & (pooled.lower_eq > -MARGIN_H4B)
    pooled["note"] = np.where(pooled.noninferior, "", M3_NOTE)
    return {"cells": cells, "pooled": pooled}


# ------------------------------------------------------------------------------------------------- H4a
H4A_ARMS = ("mvp", "target_only", "pooled", "select_on_source")
H4A_MAIN, H4A_POOLED = "mvp_vs_target_only", "mvp_vs_pooled"     # quyết định H4a chỉ theo mvp_vs_target_only; mvp_vs_pooled mô tả (QĐ97c)


def _arm_select(per: pd.DataFrame, arm: str) -> pd.DataFrame:
    if arm == "mvp":
        return per[(per.method == "mvp") & (per.gamma_rule == m2.MVP_RULE)]
    if arm == "target_only":
        return per[(per.method == "mvp") & (per.gamma_rule == "fixed") & (per.gamma == 0) & (per.prefilter_space == "target")]
    return per[per.method == arm]


def _panels(per: pd.DataFrame, genes: pd.DataFrame, n_t: str, k: int, arm: str) -> list:
    """Danh sách panel (tập gen) của nhánh qua các lần chạy; [] nếu nhánh không chạy ở (n_T, k). MVP lấy panel của dòng fixed đúng γ,
    không gian gen mà dòng m5_pos ghi; pooled, select-on-source lấy panel của chính phương pháp đó."""
    per = per.copy(); genes = genes.copy()
    per["n_t"] = per.n_t.astype(str); genes["n_t"] = genes.n_t.astype(str)
    rows = _arm_select(per[(per.n_t == n_t) & (per.k == k)], arm)
    if rows.empty:
        return []
    method = "mvp" if arm in ("mvp", "target_only") else arm
    g = genes[(genes.n_t == n_t) & (genes.k == k) & (genes.method == method)].copy()
    g["gamma_key"] = g.gamma.fillna(-1.0).astype(float)
    groups = {key: set(x.gene) for key, x in g.groupby(["split", "subset", "gamma_key", "prefilter_space"])}
    out = []
    for r in rows.sort_values(["split", "subset"]).itertuples():
        gk = -1.0 if pd.isna(r.gamma) else float(r.gamma)
        panel = groups.get((r.split, r.subset, gk, r.prefilter_space))
        if panel is None or len(panel) != k:
            raise ValueError(f"n_T = {n_t}, k = {k}: thiếu panel {arm} ở (split {r.split}, subset {r.subset})")
        out.append(sorted(panel))
    return out


def _phi(panels: list, d: int) -> dict:
    universe = sorted(set().union(*panels))
    Z = np.zeros((len(panels), d), dtype=np.int8)
    idx = {g: j for j, g in enumerate(universe)}
    for i, p in enumerate(panels):
        Z[i, [idx[g] for g in p]] = 1
    return nogueira(Z)


def h4a_cells(data: dict, genes: dict, space: dict) -> pd.DataFrame:
    """Mỗi (cohort, n_T): Φ Nogueira từng nhánh (trung bình k ∈ K_AVG), SE (trung bình SE theo k, cận trên), CI 95 % hai phía;
    delta, se = MVP − target-only (H4a); delta_vs_pooled, se_vs_pooled = MVP − pooled (mô tả; chỉ ở n_T nơi D3 chạy).
    select-on-source chỉ báo kèm (panel cố định trong cohort, Φ = 1 theo cấu tạo)."""
    rows = []
    for cohort, (per, _) in data.items():
        for n_t in _levels(per):
            stats = {}
            for arm in H4A_ARMS:
                parts = [p for p in (_panels(per, genes[cohort], n_t, k, arm) for k in K_AVG)]
                stats[arm] = [_phi(p, space[cohort]) for p in parts] if all(parts) else None
            if stats["mvp"] is None or stats["target_only"] is None:
                raise ValueError(f"{cohort} n_T = {n_t}: thiếu panel MVP hoặc target-only")
            row = {"cohort": cohort, "n_t": n_t, "k_avg": " ".join(map(str, K_AVG)), "n_runs": int(stats["mvp"][0]["M"]), "d": int(space[cohort])}
            for arm, st in stats.items():
                phi = float(np.mean([x["stability"] for x in st])) if st else np.nan
                se = float(np.mean([x["se"] for x in st])) if st else np.nan
                row.update({f"phi_{arm}": phi, f"se_{arm}": se, f"ci_low_{arm}": phi - m2.Z975 * se, f"ci_high_{arm}": phi + m2.Z975 * se})

            def diff_se(other):
                st = stats[other]
                return float(np.mean([np.sqrt(a["se"] ** 2 + b["se"] ** 2) for a, b in zip(stats["mvp"], st)])) if st else np.nan

            row.update({"delta": row["phi_mvp"] - row["phi_target_only"], "se": diff_se("target_only"),
                        "delta_vs_pooled": row["phi_mvp"] - row["phi_pooled"], "se_vs_pooled": diff_se("pooled")})
            rows.append(row)
    return pd.DataFrame(rows)


def h4a_pooled(cells: pd.DataFrame) -> pd.DataFrame:
    out = []
    sets = {**m2.POOL_SETS, "all4": tuple(cells.cohort.unique())}
    for comp, dcol, scol in ((H4A_MAIN, "delta", "se"), (H4A_POOLED, "delta_vs_pooled", "se_vs_pooled")):
        cc = cells[["cohort", "n_t", dcol, scol]].rename(columns={dcol: "delta", scol: "se"}).dropna()
        for n_t, g in cc.groupby("n_t", sort=False):
            for name, members in sets.items():
                sub = g[g.cohort.isin(members)]
                if len(sub):
                    out.append({"comparison": comp, "n_t": n_t, "pool_set": name, "cohorts": " ".join(sub.cohort), **m2.pool(sub, 0.0)})
    return pd.DataFrame(out)


def decide_h4a(pooled: pd.DataFrame) -> tuple[str, list]:
    main = pooled[(pooled.pool_set == "gpl570") & pooled.n_t.isin(H4_N_T)]
    if "comparison" in main:
        main = main[main.comparison == H4A_MAIN]
    ok = [(r.n_t, bool(r.delta_ivw >= H4A_MIN_DELTA and r.ci95_ivw_low > 0 and r.delta_eq >= H4A_MIN_DELTA and r.ci95_eq_low > 0)) for r in main.itertuples()]
    n_ok = sum(v for _, v in ok)
    outcome = "ung_ho" if ok and n_ok == len(ok) else ("ung_ho_mot_phan" if n_ok else "khong_ung_ho")
    return outcome, ok


# ------------------------------------------------------------------------------------------------- phân bố γ
def gamma_distribution(data: dict) -> pd.DataFrame:
    rows = []
    for cohort, (per, _) in data.items():
        p = per[per.gamma_rule.isin(["m5_pos", "m5_cv"])].copy()
        p["n_t"] = p.n_t.astype(str)
        for (rule, n_t, k, gamma), g in p.groupby(["gamma_rule", "n_t", "k", "gamma"]):
            rows.append({"cohort": cohort, "rule": rule, "n_t": n_t, "k": int(k), "gamma": float(gamma), "n_runs": int(len(g))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- CLI
def main(argv=None, script=__file__) -> int:
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t0 = time.time()
    main_data, report = m2.load_inputs(args.rq2, args.cohorts)
    n_gate = m2.check_gate(report)
    _, main_genes, main_cfg = load_variant(args.rq2, "RQ2_mvp", args.cohorts)
    shuf, _, shuf_cfg = load_variant(args.rq4, "RQ2_mvp_shuffled", args.cohorts)
    noihc, _, noihc_cfg = load_variant(args.rq4, "RQ2_mvp_noIHC", args.cohorts)
    if not all(c.get("source_labels_shuffled") for c in shuf_cfg.values()):
        raise SystemExit("RQ2_mvp_shuffled không ghi source_labels_shuffled = true")
    opened = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"mở kết quả RQ4 lúc {opened}", flush=True)
    space = {c: int(main_cfg[c]["gene_space_size"]) for c in args.cohorts}
    h4b = h4b_tables(shuf, args.n_boot)
    h4a_c = h4a_cells(main_data, main_genes, space)
    h4a_p = h4a_pooled(h4a_c)
    outcome, levels = decide_h4a(h4a_p)
    abl = ablation_tables(main_data, args.n_boot)
    nih = noihc_tables(main_data, noihc, args.n_boot)
    gam = gamma_distribution(main_data)
    hb = h4b["pooled"][(h4b["pooled"].pool_set == "gpl570") & h4b["pooled"].n_t.isin(H4_N_T)]
    decisions = pd.DataFrame([
        {"hypothesis": "H4a", "outcome": outcome, "family": "Δ Nogueira ≥ 0,1 và CI 95 % hai phía không chứa 0 (IVW và trọng số bằng nhau), n_T ∈ {5, 10, 20}, GPL570",
         "detail": "; ".join(f"n_T={n}: {'đạt' if v else 'không đạt'}" for n, v in levels)},
        {"hypothesis": "H4b", "outcome": "chi_bao_cao", "family": "M3: biên 0,01; không kiểm định chính thức",
         "detail": "; ".join(f"n_T={r.n_t}: Δ_ivw={r.delta_ivw:.4f}, lower_ivw={r.lower_ivw:.4f}, lower_eq={r.lower_eq:.4f}{' (' + r.note + ')' if r.note else ''}"
                             for r in hb.itertuples())}])
    out = results_dir(args.out)
    tables = {"rq4_h4b_cells": h4b["cells"], "rq4_h4b_pooled": h4b["pooled"], "rq4_h4a_cells": h4a_c, "rq4_h4a_pooled": h4a_p, "rq4_decisions": decisions,
              "rq2_ablation_cells": abl["cells"], "rq2_ablation_pooled": abl["pooled"], "rq2_noihc_cells": nih["cells"], "rq2_noihc_pooled": nih["pooled"],
              "rq2_gamma_distribution": gam}
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    cfg = {"experiment": "RQ4_tables", "rq2_input": str(args.rq2), "rq4_input": str(args.rq4),
           "rq2_input_sha256": sha256(args.rq2) if args.rq2.suffix == ".zip" else None, "rq4_input_sha256": sha256(args.rq4) if args.rq4.suffix == ".zip" else None,
           "cohorts": args.cohorts, "k_avg": list(K_AVG), "mvp_rule": m2.MVP_RULE, "n_boot": args.n_boot, "equality_rows_ok": n_gate,
           "h4b_margin": MARGIN_H4B, "h4a_min_delta": H4A_MIN_DELTA, "h4_n_t": list(H4_N_T),
           "h4a_choices": "Φ Nogueira trên 20 panel mỗi (cohort, n_T, k), d = gene_space_size; Δ trung bình k ∈ {2, 3, 5}; SE = √(SE²_MVP + SE²_TO), trung bình theo k "
                          "(bảo thủ); gộp IVW và trọng số bằng nhau; đạt khi Δ ≥ 0,1 và CI 95 % hai phía > 0 ở cả hai bản gộp. Cần GVHD xác nhận",
           "gene_space_size": space, "shuffle_seeds": {c: v.get("shuffle_seed") for c, v in shuf_cfg.items()},
           "noihc_excluded": {c: v.get("excluded_genes") for c, v in noihc_cfg.items()}, "results_opened_at_utc": opened,
           "elapsed_s": time.time() - t0, "versions": all_versions()}
    write_config(out, prov, cfg, data_files=[p for p in (args.rq2, args.rq4) if p.suffix == ".zip"])
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(decisions.to_string(index=False))
    print("ghi", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
