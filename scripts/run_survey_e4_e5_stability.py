"""E4 (độ ổn định theo tỉ lệ lấy mẫu con) và E5 (nhạy với dư thừa, nhạy với seed, chồng lấn giữa phương pháp) — survey v3.

E4: với mỗi bộ trong STABILITY_DATASETS (TCGA LUAD+LUSC, GSE37745) và tỉ lệ f ∈ {0,5; 0,7; 0,9}: rút M = 30 tập con bệnh nhân
    ĐỘC LẬP (phân tầng theo nhãn), lọc sơ bộ không dùng nhãn trong từng tập con, chọn k gen (k ∈ K_GRID) bằng từng phương pháp
    (11 phương pháp không wrapper); đo Nogueira (chính, kèm CI 95%) trên KHÔNG GIAN GEN G, Kuncheva và Jaccard trung bình (phụ);
    random-k làm sàn tham chiếu (kỳ vọng Nogueira ≈ 0).
E5a: chỉ REDUNDANCY_DATASET (GSE37745), tại f = 0,7: thêm 50 bản sao tương quan (ρ ∈ {0,5; 0,8; 0,95}) của 50 gen phương sai
    lớn nhất sau lọc sơ bộ; đo lại ổn định (thô và sau khi gộp bản sao về gen gốc) và số cặp (gốc, bản sao) cùng được chọn.
E5b: chỉ RANKING_DATASET (GSE37745), lần lặp 0 fold 0: SEED_VARIABILITY_METHODS (RF importance, stability selection; mọi k)
    và GA, BPSO (k ∈ --wrapper-k) với 5 seed; AUROC test của LR-L2 cho từng seed; Nogueira/Jaccard giữa 5 tập gen.
E5c: chồng lấn Jaccard giữa các phương pháp ở k = OVERLAP_K (10) từ kết quả pha A của E1 (selected_genes.csv), nếu đã có.

    python scripts/run_survey_e4_e5_stability.py --n-jobs 4                      # E4 (2 bộ) + E5a, E5b (GSE37745) + E5c
    python scripts/run_survey_e4_e5_stability.py --datasets gse37745 --methods f_test l1_lr --m 5
    python scripts/run_survey_e4_e5_stability.py --wrappers --n-jobs 6             # thêm GA/BPSO vào E4 (f = 0,7, k = OVERLAP_K)

Đầu ra: results/survey_v2/E4_E5/<bộ>/{subsamples.csv, e4_selected_genes.csv, e4_stability.csv, e5a_selected_genes.csv,
e5a_redundancy.csv, e5b_seed_runs.csv, e5b_seed_stability.csv, e5c_overlap_k10.csv, config.json}. Chạy lại sẽ bỏ qua
các (fraction, subsample, method) đã có trong e4_selected_genes.csv / e5a_selected_genes.csv (resume).
v3.1.6 (QĐ25): --redo-methods M ... bỏ dòng của các phương pháp này trong e4_selected_genes.csv, e4_select_times.csv,
e5a_selected_genes.csv, e5a_redundancy_per_subsample.csv (và e5b_* nếu phương pháp thuộc E5b) rồi chọn lại; E5c tính lại từ pha A.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.classify import evaluate_panel  # noqa: E402
from survey_v2.data import (add_correlated_copies, load_dataset, repeated_patient_splits, stratified_subsample,  # noqa: E402
                            variance_prefilter)
from survey_v2.methods import random_panel, select  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, previous_config, results_dir, start_run, write_config  # noqa: E402
from survey_v2.stability import kuncheva_index, mean_jaccard, nogueira, overlap_matrix, selections_to_matrix  # noqa: E402
from scripts.run_e1_classification import drop_rows_of_methods  # noqa: E402

COHORT_ID = {name: i for i, name in enumerate(C.DATASETS)}
METHOD_ID = dict(C.METHOD_ID)
NON_WRAPPER = list(C.NON_WRAPPER_METHODS)
TAG_E4, TAG_E5A, TAG_E5B = 4, 5, 6   # phần tử phân biệt trong seed


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.STABILITY_DATASETS))
    ap.add_argument("--methods", nargs="+", default=NON_WRAPPER)
    ap.add_argument("--k", nargs="+", type=int, default=list(C.K_GRID))
    ap.add_argument("--m", type=int, default=C.STABILITY_M, help="số tập con độc lập M")
    ap.add_argument("--fractions", nargs="+", type=float, default=list(C.SUBSAMPLE_FRACTIONS))
    ap.add_argument("--prefilter", type=int, default=C.PREFILTER_P0)
    ap.add_argument("--wrappers", action="store_true", help="E4: thêm GA, BPSO tại f = REDUNDANCY_FRACTION, k = OVERLAP_K")
    ap.add_argument("--wrapper-k", nargs="+", type=int, default=[C.OVERLAP_K])
    ap.add_argument("--pop", type=int, default=C.WRAPPER_POP)
    ap.add_argument("--generations", type=int, default=C.WRAPPER_GENERATIONS)
    ap.add_argument("--skip-e5a", action="store_true")
    ap.add_argument("--skip-e5b", action="store_true")
    ap.add_argument("--e1-dir", default="E1_feature_selection", help="thư mục pha A của E1 (trong results/survey_v2/) để tính E5c")
    ap.add_argument("--e1-wrapper-dir", default="E1_wrapper_feature_selection")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", default="E4_E5")
    ap.add_argument("--redo-methods", nargs="+", default=None, help="v3.1.6: bỏ dòng cũ của các phương pháp này rồi chọn lại")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------------------------------
# Một tập con: lọc sơ bộ → chọn gen bằng các phương pháp → danh sách (method, k, rank, gene)
# ---------------------------------------------------------------------------------------------------
def select_on_subset(ds, idx, methods, k_list, seed_parts, args, extra_cols: dict, rho: float | None = None):
    """Trả về (gene_rows, time_rows, redundancy_rows). Nếu rho khác None thì thêm bản sao tương quan (E5a)."""
    cid = COHORT_ID.get(ds.name, 99)
    X, y = ds.X[idx], ds.y[idx]
    gene_rows, time_rows, red_rows = [], [], []
    with threadpool_limits(1):
        pre = variance_prefilter(X, args.prefilter)
        Xp, genes = X[:, pre], ds.genes[pre]
        src_idx = copy_idx = None
        if rho is not None:
            rng = np.random.default_rng([cid, TAG_E5A, *seed_parts, int(round(rho * 100))])
            Xp, genes, src_idx, copy_idx = add_correlated_copies(Xp, genes, rho, C.REDUNDANCY_N_COPIES, rng)
        p0 = Xp.shape[1]
        for method in methods:
            ks = [k for k in k_list if k <= p0]
            t0 = time.perf_counter()
            if method == "random_k":
                panels = {k: random_panel(p0, k, np.random.default_rng([cid, *seed_parts, k])) for k in ks}
                info = {}
            else:
                try:
                    sel = select(method, Xp, y, ks, seed=C.seed_for(cid, *seed_parts, METHOD_ID.get(method, 50)),
                                 n_jobs=1, wrapper_kwargs={"pop": args.pop, "generations": args.generations})
                except ImportError as exc:
                    print(f"[{ds.name}] {method}: bỏ qua ({exc})", flush=True)
                    continue
                except ValueError as exc:
                    if "KTSP_MAX_P" not in str(exc):
                        raise
                    # v3.1.5: không để một phương pháp làm hỏng cả phiên; ghi nhận và chạy tiếp (E5a với p0 + bản sao > cap)
                    print(f"[{ds.name}] {method}: bỏ qua ở p = {p0} ({exc})", flush=True)
                    time_rows.append({**extra_cols, "method": method, "n": len(idx), "p0": p0, "select_time_s": float("nan"),
                                      "skipped": str(exc)})
                    continue
                panels, info = sel.panels, sel.info
            dt = time.perf_counter() - t0
            time_rows.append({**extra_cols, "method": method, "n": len(idx), "p0": p0, "select_time_s": dt})
            for k in ks:
                if panels.get(k) is None:          # k-TSP ở k lẻ
                    continue
                sel_idx = np.asarray(panels[k])
                for rank, j in enumerate(sel_idx):
                    gene_rows.append({**extra_cols, "method": method, "k": int(k), "rank": rank + 1, "gene": genes[j]})
                if rho is not None:
                    chosen = set(sel_idx.tolist())
                    src_set, copy_set = set(src_idx.tolist()), set(copy_idx.tolist())
                    pairs_both = sum(1 for s, c in zip(src_idx, copy_idx) if s in chosen and c in chosen)
                    red_rows.append({**extra_cols, "method": method, "k": int(k),
                                     "n_source_selected": len(chosen & src_set), "n_copy_selected": len(chosen & copy_set),
                                     "pairs_both_selected": int(pairs_both),
                                     "n_filled": int(info.get("n_filled", {}).get(int(k), 0))})
    return gene_rows, time_rows, red_rows


def append_csv(path: Path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=not path.exists(), encoding="utf-8-sig")


def done_triples(path: Path, cols):
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=cols).drop_duplicates()
    return set(map(tuple, df[cols].itertuples(index=False, name=None)))


# ---------------------------------------------------------------------------------------------------
# Tính chỉ số ổn định từ bảng gen đã chọn
# ---------------------------------------------------------------------------------------------------
def _merge_copies(gene: str) -> str:
    return gene.split("__copy_rho")[0]


def stability_table(genes_df: pd.DataFrame, universe, group_cols, merge_copies: bool = False) -> pd.DataFrame:
    """Với mỗi nhóm (group_cols + method + k): Z (M × d) trên `universe`, Nogueira + CI, Kuncheva, Jaccard."""
    rows = []
    uni = list(universe)
    d = len(uni)
    for key, g in genes_df.groupby([*group_cols, "method", "k"], sort=True):
        sets = [list(s["gene"].map(_merge_copies) if merge_copies else s["gene"]) for _, s in g.groupby("subsample")]
        sets = [list(dict.fromkeys(s)) for s in sets]   # bỏ trùng (sau khi gộp bản sao)
        Z = selections_to_matrix(sets, uni)
        nog = nogueira(Z)
        same_size = len({len(s) for s in sets}) == 1
        row = dict(zip([*group_cols, "method", "k"], key))
        row.update({"M": len(sets), "d": d, "k_mean": float(np.mean([len(s) for s in sets])),
                    "nogueira": nog["stability"], "nogueira_se": nog["se"], "nogueira_ci_low": nog["ci_low"],
                    "nogueira_ci_high": nog["ci_high"],
                    "kuncheva": kuncheva_index(sets, d) if same_size else np.nan,
                    "jaccard_mean": mean_jaccard(sets)})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------
def run_e4_e5a(ds, out: Path, args) -> dict:
    cid = COHORT_ID.get(ds.name, 99)
    methods = list(args.methods) + ["random_k"]
    frac_list = list(args.fractions)
    # ---- tập con độc lập (chung cho E4 và E5a) -----------------------------------------------------
    subsets = {}
    sub_rows = []
    for fi, f in enumerate(frac_list):
        for i in range(args.m):
            idx = stratified_subsample(ds.y, f, np.random.default_rng([cid, TAG_E4, fi, i]))
            subsets[(f, i)] = idx
            sub_rows += [{"cohort": ds.name, "fraction": f, "subsample": i, "sample_id": s, "label": int(ds.y[j])}
                         for j, s in zip(idx, ds.sample_ids[idx])]
    pd.DataFrame(sub_rows).to_csv(out / "subsamples.csv", index=False)

    # ---- E4 ---------------------------------------------------------------------------------------------
    e4_genes = out / "e4_selected_genes.csv"
    done = done_triples(e4_genes, ["fraction", "subsample", "method"])
    jobs = []
    for fi, f in enumerate(frac_list):
        for i in range(args.m):
            todo = [m for m in methods if (f, i, m) not in done]
            if todo:
                jobs.append((f, fi, i, todo))
    if args.wrappers:
        fw = C.REDUNDANCY_FRACTION
        if fw in frac_list:
            fi = frac_list.index(fw)
            for i in range(args.m):
                todo = [m for m in C.WRAPPER_METHODS if (fw, i, m) not in done]
                if todo:
                    jobs.append((fw, fi, i, todo))
    print(f"[{ds.name}] E4: {len(jobs)} (fraction, subsample) cần chạy", flush=True)

    def job(f, fi, i, todo):
        ks = args.k if not set(todo) <= set(C.WRAPPER_METHODS) else args.wrapper_k
        cols = {"cohort": ds.name, "fraction": f, "subsample": i}
        return select_on_subset(ds, subsets[(f, i)], todo, ks, (TAG_E4, fi, i), args, cols)

    for res in Parallel(n_jobs=args.n_jobs, backend="loky", return_as="generator")(delayed(job)(*j) for j in jobs):
        gene_rows, time_rows, _ = res
        append_csv(e4_genes, gene_rows)
        append_csv(out / "e4_select_times.csv", time_rows)
    e4 = pd.read_csv(e4_genes)
    stab = stability_table(e4, ds.genes, ["cohort", "fraction"])
    stab.to_csv(out / "e4_stability.csv", index=False, encoding="utf-8-sig")
    print(f"[{ds.name}] E4 xong: {len(stab)} dòng ổn định", flush=True)

    # ---- E5a --------------------------------------------------------------------------------------------
    n_e5a_rows = 0
    if not args.skip_e5a and ds.name != C.REDUNDANCY_DATASET:
        print(f"[{ds.name}] E5a chỉ chạy trên {C.REDUNDANCY_DATASET}; bỏ qua", flush=True)
    elif not args.skip_e5a:
        fw = C.REDUNDANCY_FRACTION
        if fw not in frac_list:
            print(f"[{ds.name}] E5a bỏ qua: --fractions không chứa {fw}", flush=True)
        else:
            fi = frac_list.index(fw)
            e5a_genes = out / "e5a_selected_genes.csv"
            done5 = done_triples(e5a_genes, ["rho", "subsample", "method"])
            jobs5 = []
            for rho in C.REDUNDANCY_RHO:
                for i in range(args.m):
                    todo = [m for m in methods if (rho, i, m) not in done5]
                    if todo:
                        jobs5.append((rho, i, todo))
            print(f"[{ds.name}] E5a: {len(jobs5)} (rho, subsample) cần chạy", flush=True)

            def job5(rho, i, todo):
                cols = {"cohort": ds.name, "fraction": fw, "rho": rho, "subsample": i}
                return select_on_subset(ds, subsets[(fw, i)], todo, args.k, (TAG_E4, fi, i), args, cols, rho=rho)

            for res in Parallel(n_jobs=args.n_jobs, backend="loky", return_as="generator")(delayed(job5)(*j) for j in jobs5):
                gene_rows, time_rows, red_rows = res
                append_csv(e5a_genes, gene_rows)
                append_csv(out / "e5a_redundancy_per_subsample.csv", red_rows)
            e5a = pd.read_csv(e5a_genes)
            tabs = []
            for rho, g in e5a.groupby("rho"):
                uni_rho = list(ds.genes) + [f"{x}__copy_rho{rho}" for x in ds.genes]  # bản sao có thể của mọi gen
                raw = stability_table(g, uni_rho, ["cohort", "fraction", "rho"])
                merged = stability_table(g, ds.genes, ["cohort", "fraction", "rho"], merge_copies=True)
                merged = merged.rename(columns={c: f"{c}_merged" for c in merged.columns
                                                if c not in ("cohort", "fraction", "rho", "method", "k")})
                tabs.append(raw.merge(merged, on=["cohort", "fraction", "rho", "method", "k"]))
            red = pd.read_csv(out / "e5a_redundancy_per_subsample.csv")
            red_mean = (red.groupby(["cohort", "fraction", "rho", "method", "k"], as_index=False)
                        [["n_source_selected", "n_copy_selected", "pairs_both_selected", "n_filled"]].mean())
            e5a_tab = pd.concat(tabs, ignore_index=True).merge(red_mean, on=["cohort", "fraction", "rho", "method", "k"])
            base = stab[stab.fraction == fw][["cohort", "method", "k", "nogueira"]].rename(columns={"nogueira": "nogueira_e4_same_fraction"})
            e5a_tab = e5a_tab.merge(base, on=["cohort", "method", "k"], how="left")
            e5a_tab["delta_nogueira_vs_e4"] = e5a_tab["nogueira"] - e5a_tab["nogueira_e4_same_fraction"]
            e5a_tab.to_csv(out / "e5a_redundancy.csv", index=False, encoding="utf-8-sig")
            n_e5a_rows = len(e5a_tab)
            print(f"[{ds.name}] E5a xong: {n_e5a_rows} dòng", flush=True)
    return {"n_subsamples": args.m, "fractions": frac_list, "e4_rows": int(len(stab)), "e5a_rows": n_e5a_rows}


# ---------------------------------------------------------------------------------------------------
def run_e5b(ds, out: Path, args) -> dict:
    """GA, BPSO (k ∈ wrapper_k) và RF importance (k ∈ --k) với 5 seed trên lần lặp 0, fold 0."""
    cid = COHORT_ID.get(ds.name, 99)
    repeat, seed, fold, tr, te = repeated_patient_splits(ds.y, ds.groups)[0]
    Xtr, ytr, Xte, yte = ds.X[tr], ds.y[tr], ds.X[te], ds.y[te]
    plan = [(m, list(args.k)) for m in C.SEED_VARIABILITY_METHODS] + [(m, list(args.wrapper_k)) for m in C.WRAPPER_METHODS]
    runs = out / "e5b_seed_runs.csv"
    done = done_triples(runs, ["method", "seed_id"]) if runs.exists() else set()
    with threadpool_limits(1):
        pre = variance_prefilter(Xtr, args.prefilter)
        Xtr_p, Xte_p = Xtr[:, pre], Xte[:, pre]

    def one(method, ks, s):
        with threadpool_limits(1):
            t0 = time.perf_counter()
            sel = select(method, Xtr_p, ytr, ks, seed=C.seed_for(cid, TAG_E5B, s, METHOD_ID[method]), n_jobs=1,
                         wrapper_kwargs={"pop": args.pop, "generations": args.generations})
            dt = time.perf_counter() - t0
            rows, gene_rows = [], []
            for k in ks:
                idx = sel.panels.get(k)
                if idx is None:
                    continue
                r = evaluate_panel(Xtr_p[:, idx], ytr, Xte_p[:, idx], yte, classifier="lr_l2",
                                   seed=C.seed_for(cid, seed, fold, 1000 + k))
                rows.append({"cohort": ds.name, "repeat": repeat, "fold": fold, "method": method, "seed_id": s, "k": int(k),
                             "auroc": r.auroc, "bacc": r.bacc, "mcc": r.mcc, "select_time_s": dt,
                             "n_eval": sel.info.get("n_eval", {}).get(k), "best_fitness": sel.info.get("best_fitness", {}).get(k)})
                gene_rows += [{"cohort": ds.name, "method": method, "seed_id": s, "k": int(k), "rank": r_ + 1, "gene": ds.genes[pre[j]]}
                              for r_, j in enumerate(idx)]
            return rows, gene_rows

    jobs = [(m, ks, s) for m, ks in plan for s in C.SEED_VARIABILITY_SEEDS if (m, s) not in done]
    print(f"[{ds.name}] E5b: {len(jobs)} (method, seed) cần chạy", flush=True)
    for rows, gene_rows in Parallel(n_jobs=args.n_jobs, backend="loky", return_as="generator")(delayed(one)(*j) for j in jobs):
        append_csv(runs, rows)
        append_csv(out / "e5b_selected_genes.csv", gene_rows)
    df = pd.read_csv(runs)
    genes = pd.read_csv(out / "e5b_selected_genes.csv").rename(columns={"seed_id": "subsample"})
    stab = stability_table(genes, ds.genes, ["cohort"]).rename(columns={"M": "n_seeds"})
    perf = (df.groupby(["cohort", "method", "k"], as_index=False)
            .agg(auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"), auroc_min=("auroc", "min"),
                 auroc_max=("auroc", "max"), bacc_mean=("bacc", "mean"), mcc_mean=("mcc", "mean"),
                 select_time_mean_s=("select_time_s", "mean"), n_eval_mean=("n_eval", "mean")))
    stab.merge(perf, on=["cohort", "method", "k"]).to_csv(out / "e5b_seed_stability.csv", index=False, encoding="utf-8-sig")
    return {"split": {"repeat": repeat, "seed": seed, "fold": fold}, "seeds": list(C.SEED_VARIABILITY_SEEDS),
            "wrapper_k": list(args.wrapper_k)}


# ---------------------------------------------------------------------------------------------------
def run_e5c(ds, out: Path, e1_dir: Path, e1_wrapper_dir: Path | None = None) -> dict:
    frames = []
    for d in (e1_dir, e1_wrapper_dir):
        if d is None:
            continue
        f = d / ds.name / "selected_genes.csv"
        if f.exists():
            try:
                frames.append(pd.read_csv(f))
            except pd.errors.EmptyDataError:
                pass
    if not frames:
        print(f"[{ds.name}] E5c bỏ qua: chưa có selected_genes.csv của pha A trong {e1_dir}", flush=True)
        return {"status": "skipped", "reason": f"missing {e1_dir / ds.name / 'selected_genes.csv'}"}
    g = pd.concat(frames, ignore_index=True)
    g = g[(g.k == C.OVERLAP_K) & (~g.method.isin(["ihc_2", "ihc_4"])) & (~g.method.str.startswith("panel_"))]
    if g.empty:
        print(f"[{ds.name}] E5c bỏ qua: {f} không có k = {C.OVERLAP_K}", flush=True)
        return {"status": "skipped", "reason": f"no k={C.OVERLAP_K} in {f}"}
    folds = sorted(set(map(tuple, g[["repeat", "fold"]].drop_duplicates().itertuples(index=False, name=None))))
    methods = sorted(g.method.unique())
    panels = {}
    for m in methods:
        gm = g[g.method == m]
        have = set(map(tuple, gm[["repeat", "fold"]].drop_duplicates().itertuples(index=False, name=None)))
        if not have:
            continue
        panels[m] = [set(gm[(gm.repeat == r) & (gm.fold == fo)].gene) if (r, fo) in have else None for r, fo in folds]
    # chỉ so trên các fold mà CẢ HAI phương pháp đều có (wrapper chỉ 3 lần lặp)
    names = list(panels)
    M = np.full((len(names), len(names)), np.nan)
    n_pairs = np.zeros((len(names), len(names)), dtype=int)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            vals = [len(x & y) / len(x | y) for x, y in zip(panels[a], panels[b]) if x is not None and y is not None and (x | y)]
            M[i, j] = float(np.mean(vals)) if vals else np.nan
            n_pairs[i, j] = len(vals)
    mat = pd.DataFrame(M, index=names, columns=names)
    mat.index.name = "method"
    mat.to_csv(out / f"e5c_overlap_k{C.OVERLAP_K}.csv", encoding="utf-8-sig")
    pd.DataFrame(n_pairs, index=names, columns=names).to_csv(out / f"e5c_overlap_k{C.OVERLAP_K}_nfolds.csv")
    return {"status": "ok", "methods": names, "k": C.OVERLAP_K, "n_folds_total": len(folds)}


# ---------------------------------------------------------------------------------------------------
def main(argv=None, script=__file__) -> None:
    args = parse_args(argv)
    prov = start_run(script, params=vars(args))
    t_start = time.time()
    e1_dir = DATA_ROOT.joinpath(*C.RESULT_DIR, args.e1_dir)
    e1_wrapper_dir = DATA_ROOT.joinpath(*C.RESULT_DIR, args.e1_wrapper_dir)
    for name in args.datasets:
        ds = load_dataset(DATA_ROOT, name)
        out = results_dir(args.out, name)
        print(f"[{name}] {ds.n} mẫu × {ds.p} gen", flush=True)
        prev_cfg = previous_config(out)
        if args.redo_methods:
            n_drop = drop_rows_of_methods([out / f for f in ("e4_selected_genes.csv", "e4_select_times.csv", "e5a_selected_genes.csv",
                                                             "e5a_redundancy_per_subsample.csv", "e5b_seed_runs.csv", "e5b_selected_genes.csv")],
                                          args.redo_methods)
            print(f"[{name}] --redo-methods {args.redo_methods}: đã bỏ {n_drop} dòng cũ, sẽ chọn lại", flush=True)
        info = {"experiment": "E4_E5", "dataset": name, "n_samples": ds.n, "n_genes": ds.p, "k_grid": args.k,
                "methods": args.methods + ["random_k"], "prefilter_p0": args.prefilter,
                "redo_methods": list(args.redo_methods or []),
                "rows_reused_from_commit": (prev_cfg or {}).get("commit_hash") if args.redo_methods else None,
                "subsample_seed": "default_rng([cohort_id, 4, fraction_index, subsample])",
                "stability_universe": "gene space G of the dataset (prefilter is inside each subsample)", "gene_space": ds.notes,
                "role": C.DATASETS.get(name),
                "nogueira_ci": "asymptotic 95% CI (Nogueira et al. 2018), valid for independent subsamples"}
        info["e4_e5a"] = run_e4_e5a(ds, out, args)
        if not args.skip_e5b and name == C.RANKING_DATASET:
            info["e5b"] = run_e5b(ds, out, args)
        info["e5c"] = run_e5c(ds, out, e1_dir, e1_wrapper_dir)
        info["versions"] = all_versions()
        info["elapsed_s"] = time.time() - t_start
        write_config(out, prov, info, data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv"])
        print(f"[{name}] xong, ghi vào {out}", flush=True)


if __name__ == "__main__":
    main()
