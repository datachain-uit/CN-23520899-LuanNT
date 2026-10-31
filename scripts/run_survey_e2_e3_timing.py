"""E2 (chi phí tại một điểm) và E3 (đường cong độ phức tạp theo n, p; wrapper theo số lần đánh giá) — survey v3.

Chạy trên MỘT máy cố định, BLAS 1 luồng (script tự đặt OMP/OPENBLAS/MKL_NUM_THREADS = 1 trước khi import
numpy). Dữ liệu: TCGA LUAD + LUSC (TIMING_DATASET; n ≈ 814 train, p = 19.938); phần train của repeat 0, fold 0 của CV 5×5.

E2: n = toàn bộ phần train, p = 5.000 (lọc phương sai), k ∈ TIMING_K = {10, 50}; wall time, CPU time, bộ nhớ đỉnh; trung vị 3 lần.
E3: theo n ∈ {100, 200, 400, toàn bộ} (p = 5.000) và theo p ∈ {1.000, 2.000, 5.000, 10.000, toàn bộ} (n toàn bộ) ở k = TIMING_K[0];
    số mũ log–log của thời gian theo n và theo p (so với Bảng 5 của bản v1); wrapper theo (quần thể × thế hệ)
    ∈ {10×20, 30×50} (--wrappers). k-TSP chỉ đo tới p = KTSP_MAX_P (O(n p^2)); abess bỏ qua nếu chưa cài.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_survey_e2_e3_timing.py
    python scripts/run_survey_e2_e3_timing.py --methods f_test mrmr --repeats 1      # chạy thử nhanh
    python scripts/run_survey_e2_e3_timing.py --wrappers                              # thêm GA, BPSO theo N_eval

v3.1 (QĐ18, Y19 — Review 29/10): (1) hiệu chuẩn giữa phiên: mỗi lần chạy đo lại CALIBRATION_METHODS ở điểm E2 và ghi
calibration.csv; với --calibration-reference <calibration.csv của phiên tham chiếu>, script tính độ lệch wall time từng mốc và
DỪNG (mã thoát 3) nếu có mốc lệch >= CALIBRATION_MAX_DRIFT (bỏ qua bằng --allow-drift; khi đó config ghi calibration_ok = false
và kết quả không được trích). (2) --isolate-memory: bộ nhớ đỉnh E2 đo trong tiến trình con mới cho từng (phương pháp, k)
(cột maxrss_isolated_mb); peak_mb (tracemalloc) chỉ còn là mô tả [bỏ hẳn ở v3.1.6].

v3.1.4 (QĐ23): --skip-e2 cho các phiên E3 chia phần (E2 chính thức = phiên tham chiếu, tất cả phương pháp trong một phiên);
C.E3_MAX_P bỏ điểm p quá lớn cho phương pháp quá chậm (elastic_net: tới 10.000).

v3.1.6 (QĐ27, Review 31/10): hai lỗi đo của các phiên 29–30/10 — (1) tracemalloc bật trong lúc đo làm wall time tăng
tuỳ phương pháp (ReliefF ×5–6, MI ×3, RF ×2,5), (2) bộ nhớ tiến trình con kế thừa đỉnh RSS của tiến trình cha — đã sửa trong
survey_v2/timing.py; E2_E3_ref và E3 p1..p3 PHẢI chạy lại ở bản này (kết quả cũ chuyển vào 04_Luu-vet). Cột peak_mb và
maxrss_mb không còn; bộ nhớ đỉnh chỉ có với --isolate-memory (maxrss_isolated_mb, nguồn VmHWM). e3_wrapper_evals.csv chỉ
được ghi khi có dòng (tệp rỗng làm merge-e2e3 lỗi).

Đầu ra: results/survey_v2/E2_E3/{e2_cost_point.csv, e3_scaling_n.csv, e3_scaling_p.csv, e3_wrapper_evals.csv,
e3_exponents.csv, calibration.csv, calibration_check.json, machine_config.json, config.json}.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import load_dataset, repeated_patient_splits  # noqa: E402
from survey_v2.methods import select  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402
from survey_v2.timing import empirical_exponent, machine_info, measure, measure_isolated, subsample_rows, top_variance_columns  # noqa: E402

NON_WRAPPER = list(C.NON_WRAPPER_METHODS)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=C.TIMING_DATASET)
    ap.add_argument("--methods", nargs="+", default=NON_WRAPPER)
    ap.add_argument("--k", nargs="+", type=int, default=list(C.TIMING_K), help="E2 đo ở mọi k; E3 dùng k đầu tiên")
    ap.add_argument("--repeats", type=int, default=C.TIMING_REPEATS, help="số lần đo, lấy trung vị")
    ap.add_argument("--n-grid", nargs="+", default=list(C.E3_N_GRID), help="dùng 'all' cho toàn bộ phần train")
    ap.add_argument("--p-grid", nargs="+", default=list(C.E3_P_GRID), help="dùng 'all' cho toàn bộ gen")
    ap.add_argument("--wrappers", action="store_true", help="đo GA, BPSO theo ngân sách (quần thể × thế hệ)")
    ap.add_argument("--skip-e3", action="store_true")
    ap.add_argument("--skip-e2", action="store_true",
                    help="chỉ hiệu chuẩn + E3 (dùng cho các phiên E3 chia phần — QĐ18; E2 chính thức lấy từ phiên tham chiếu)")
    ap.add_argument("--out", default="E2_E3")
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false",
                    help="bỏ bước hiệu chuẩn (chỉ cho chạy thử; phiên chính thức luôn hiệu chuẩn — QĐ18)")
    ap.add_argument("--calibration-reference", type=Path, default=None,
                    help="calibration.csv của phiên tham chiếu; độ lệch wall time >= CALIBRATION_MAX_DRIFT -> dừng")
    ap.add_argument("--allow-drift", action="store_true", help="không dừng khi lệch (config ghi calibration_ok = false)")
    ap.add_argument("--isolate-memory", action="store_true",
                    help="E2: đo bộ nhớ đỉnh trong tiến trình con mới cho từng (phương pháp, k) — Y19")
    ap.set_defaults(calibrate=True)
    return ap.parse_args()


def run_calibration(args, out, seed, X2, ytr, k_e3) -> dict:
    """QĐ18: đo CALIBRATION_METHODS ở điểm E2; so với phiên tham chiếu nếu có. Trả về dict ghi vào config."""
    rows = []
    for m in C.CALIBRATION_METHODS:
        r = safe_measure(m, seed, X2, ytr, k_e3, args.repeats)
        if r is None:
            continue
        rows.append({"method": m, "n": X2.shape[0], "p": X2.shape[1], "k": k_e3, "wall_s": r["wall_s"], "cpu_s": r["cpu_s"],
                     "wall_s_all": r["wall_s_all"]})
        print(f"[hiệu chuẩn] {m}: wall {r['wall_s']:.3f}s", flush=True)
    cal = pd.DataFrame(rows)
    cal.to_csv(out / "calibration.csv", index=False, encoding="utf-8-sig")
    check = {"methods": list(C.CALIBRATION_METHODS), "max_drift_allowed": C.CALIBRATION_MAX_DRIFT, "reference": None,
             "drift": {}, "max_drift": None, "calibration_ok": None}
    if args.calibration_reference is not None:
        ref = pd.read_csv(args.calibration_reference).set_index("method")
        check["reference"] = str(args.calibration_reference)
        for r in rows:
            if r["method"] in ref.index and float(ref.loc[r["method"], "wall_s"]) > 0:
                check["drift"][r["method"]] = abs(r["wall_s"] - float(ref.loc[r["method"], "wall_s"])) / float(ref.loc[r["method"], "wall_s"])
        if check["drift"]:
            check["max_drift"] = max(check["drift"].values())
            check["calibration_ok"] = bool(check["max_drift"] < C.CALIBRATION_MAX_DRIFT)
        print(f"[hiệu chuẩn] lệch so với phiên tham chiếu: {check['drift']} -> ok = {check['calibration_ok']}", flush=True)
    (out / "calibration_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if check["calibration_ok"] is False and not args.allow_drift:
        print(f"[hiệu chuẩn] DỪNG: độ lệch {check['max_drift']:.1%} >= {C.CALIBRATION_MAX_DRIFT:.0%} (QĐ18). "
              "Chạy lại ở phiên khác hoặc dùng --allow-drift (kết quả khi đó không được trích).", file=sys.stderr, flush=True)
        raise SystemExit(3)
    return check


def _grid(values):
    out = []
    for v in values:
        out.append(None if v in (None, "all", "None") else int(v))
    return out


def selector(method: str, seed: int, pop=None, generations=None):
    def run(X, y, k):
        kw = {"pop": pop, "generations": generations} if pop else None
        return select(method, X, y, [k], seed=seed, n_jobs=1, wrapper_kwargs=kw).panels[k]
    return run


def safe_measure(m, seed, X, y, k, repeats, pop=None, generations=None):
    """measure() nhưng trả về None (ghi NaN) khi phương pháp không chạy được ở điểm này (abess chưa cài; k-TSP p quá lớn)."""
    try:
        return measure(selector(m, seed, pop, generations), X, y, k, repeats=repeats)
    except ImportError as exc:
        print(f"  {m}: bỏ qua ({exc})", flush=True)
    except ValueError as exc:
        if "KTSP_MAX_P" in str(exc):
            print(f"  {m}: bỏ qua ở p = {X.shape[1]} ({exc})", flush=True)
        else:
            raise
    return None


def main() -> None:
    args = parse_args()
    prov = start_run(__file__, params=vars(args))
    out = results_dir(args.out)
    ds = load_dataset(DATA_ROOT, args.dataset)
    repeat, seed, fold, tr, _ = repeated_patient_splits(ds.y, ds.groups)[0]
    print(f"[{args.dataset}] {ds.n} mẫu × {ds.p} gen; train fold 0 = {len(tr)}", flush=True)
    Xtr, ytr = ds.X[tr], ds.y[tr]
    rng = np.random.default_rng([7, seed, fold])
    minfo = machine_info()
    (out / "machine_config.json").write_text(json.dumps(minfo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Máy:", minfo.get("cpu_model", minfo.get("processor")), "| RAM GB:", minfo.get("ram_gb"), flush=True)
    t_all = time.time()
    with threadpool_limits(1):
        # ---- E2: điểm cố định ---------------------------------------------------------------------
        p_fixed = min(C.PREFILTER_P0, Xtr.shape[1])
        cols = top_variance_columns(Xtr, p_fixed)
        X2 = Xtr[:, cols]
        rows = []
        k_e3 = int(args.k[0])
        calibration = run_calibration(args, out, seed, X2, ytr, k_e3) if args.calibrate else None
        baseline = measure_isolated(X2, ytr, k_e3, "none", seed) if args.isolate_memory else None
        if baseline is not None:
            baseline = {"maxrss_mb": baseline["maxrss_child_mb"], "peak_source": baseline.get("peak_source")}
            print(f"[bộ nhớ] mức nền của tiến trình con: {baseline['maxrss_mb']:.0f} MB ({baseline.get('peak_source')})", flush=True)
        for k in ([] if args.skip_e2 else args.k):
            for m in args.methods:
                print(f"E2 {m} k={k} ...", end=" ", flush=True)
                r = safe_measure(m, seed, X2, ytr, int(k), args.repeats)
                if r is None:
                    continue
                if args.isolate_memory:
                    r.update(measure_isolated(X2, ytr, int(k), m, seed, baseline=baseline))
                rows.append({"method": m, "family": C.METHOD_FAMILY.get(m, "other"), "n": X2.shape[0], "p": X2.shape[1], "k": int(k), **r})
                print(f"wall {r['wall_s']:.2f}s cpu {r['cpu_s']:.2f}s"
                      + (f" maxrss_isolated {r.get('maxrss_isolated_mb', float('nan')):.0f}MB" if args.isolate_memory else ""), flush=True)
            if args.wrappers:
                for m in C.WRAPPER_METHODS:
                    pop, gen = C.WRAPPER_POP, C.WRAPPER_GENERATIONS
                    print(f"E2 {m} ({pop}x{gen}) k={k} ...", end=" ", flush=True)
                    r = safe_measure(m, seed, X2, ytr, int(k), args.repeats, pop, gen)
                    if r is None:
                        continue
                    if args.isolate_memory:
                        r.update(measure_isolated(X2, ytr, int(k), m, seed, pop, gen, baseline=baseline))
                    rows.append({"method": m, "family": "wrapper", "n": X2.shape[0], "p": X2.shape[1], "k": int(k),
                                 "pop": pop, "generations": gen, **r})
                    print(f"wall {r['wall_s']:.1f}s", flush=True)
        if not args.skip_e2:
            pd.DataFrame(rows).to_csv(out / "e2_cost_point.csv", index=False, encoding="utf-8-sig")

        if not args.skip_e3:
            # ---- E3 theo n (p = 5.000) ---------------------------------------------------------------
            rows_n = []
            for n in _grid(args.n_grid):
                idx = subsample_rows(ytr, n, rng)
                for m in args.methods:
                    r = safe_measure(m, seed, X2[idx], ytr[idx], k_e3, args.repeats)
                    if r is None:
                        continue
                    rows_n.append({"method": m, "family": C.METHOD_FAMILY.get(m, "other"), "n": len(idx), "p": X2.shape[1], "k": k_e3, **r})
                    print(f"E3 n={len(idx):4d} {m:20s} wall {r['wall_s']:.2f}s", flush=True)
            pd.DataFrame(rows_n).to_csv(out / "e3_scaling_n.csv", index=False, encoding="utf-8-sig")
            # ---- E3 theo p (n toàn bộ) ---------------------------------------------------------------
            rows_p = []
            for p in _grid(args.p_grid):
                cols_p = top_variance_columns(Xtr, p)
                Xp = Xtr[:, cols_p]
                for m in args.methods:
                    if Xp.shape[1] > C.E3_MAX_P.get(m, float("inf")):      # QĐ23: bỏ điểm p quá lớn cho phương pháp quá chậm
                        print(f"E3 p={Xp.shape[1]:6d} {m:20s} bỏ qua (E3_MAX_P = {C.E3_MAX_P[m]})", flush=True)
                        continue
                    r = safe_measure(m, seed, Xp, ytr, k_e3, args.repeats)
                    if r is None:
                        continue
                    rows_p.append({"method": m, "family": C.METHOD_FAMILY.get(m, "other"), "n": Xp.shape[0], "p": Xp.shape[1], "k": k_e3, **r})
                    print(f"E3 p={Xp.shape[1]:6d} {m:20s} wall {r['wall_s']:.2f}s", flush=True)
            pd.DataFrame(rows_p).to_csv(out / "e3_scaling_p.csv", index=False, encoding="utf-8-sig")
            # ---- E3 wrapper theo số lần đánh giá ------------------------------------------------------
            rows_w = []
            if args.wrappers:
                for m in C.WRAPPER_METHODS:
                    for pop, gen in C.E3_WRAPPER_BUDGETS:
                        r = safe_measure(m, seed, X2, ytr, k_e3, args.repeats, pop, gen)
                        if r is None:
                            continue
                        rows_w.append({"method": m, "pop": pop, "generations": gen, "n_eval_nominal": pop * gen,
                                       "n": X2.shape[0], "p": X2.shape[1], "k": k_e3, **r})
                        print(f"E3 {m} {pop}x{gen} wall {r['wall_s']:.1f}s", flush=True)
            if rows_w:                                   # v3.1.6: không ghi tệp rỗng (merge-e2e3 đọc lỗi)
                pd.DataFrame(rows_w).to_csv(out / "e3_wrapper_evals.csv", index=False, encoding="utf-8-sig")
            # ---- số mũ log–log ---------------------------------------------------------------------------
            exps = []
            dn, dp = pd.DataFrame(rows_n), pd.DataFrame(rows_p)
            for m in args.methods:
                gn, gp = dn[dn.method == m] if len(dn) else dn, dp[dp.method == m] if len(dp) else dp
                if len(gn) == 0 and len(gp) == 0:
                    continue
                gp_large = gp[gp.p >= 2000]  # dải lớn: tránh chi phí cố định làm số mũ thấp (code I)
                exps.append({"method": m, "family": C.METHOD_FAMILY.get(m, "other"),
                             "alpha_n": empirical_exponent(gn, "n"), "n_range": f"{gn.n.min()}-{gn.n.max()}",
                             "alpha_p": empirical_exponent(gp, "p"), "alpha_p_large": empirical_exponent(gp_large, "p"),
                             "p_range": f"{gp.p.min()}-{gp.p.max()}",
                             "wall_spread_max_rel": float(max([(max(w) - min(w)) / max(np.median(w), 1e-9)
                                                              for w in gn.wall_s_all.tolist() + gp.wall_s_all.tolist()] or [np.nan]))})
            pd.DataFrame(exps).to_csv(out / "e3_exponents.csv", index=False, encoding="utf-8-sig")
    cfg = {"experiment": "E2_E3", "dataset": args.dataset, "split": {"repeat": repeat, "seed": seed, "fold": fold,
           "n_train": int(len(tr))}, "k": args.k, "k_e3": int(args.k[0]), "p_fixed": int(p_fixed), "repeats": args.repeats,
           "n_grid": args.n_grid, "p_grid": args.p_grid, "methods": args.methods, "wrappers": args.wrappers,
           "blas_threads": 1, "machine": minfo, "versions": all_versions(), "elapsed_s": time.time() - t_all,
           "calibration": calibration, "calibration_ok": (calibration or {}).get("calibration_ok"),
           "isolate_memory": args.isolate_memory, "skip_e2": args.skip_e2, "skip_e3": args.skip_e3, "e3_max_p": dict(C.E3_MAX_P),
           "peak_source": (baseline or {}).get("peak_source"), "timing_tracemalloc": False,
           "memory_note": ("maxrss_isolated_mb = đỉnh RSS (VmHWM trên Linux, v3.1.6/QĐ27) của tiến trình con chạy một lần chọn gen trừ mức nền "
                           "(bộ nhớ đỉnh dùng cho E2)") if args.isolate_memory
                          else "không đo bộ nhớ (chạy với --isolate-memory để có maxrss_isolated_mb)",
           "timing_note": "wall/cpu = trung vị repeats lần sau một lần khởi động; không tracemalloc (v3.1.6, QĐ27)",
           "note": "thời gian chỉ gồm bước chọn gen (không gồm huấn luyện bộ phân loại sau chọn)"}
    write_config(out, prov, cfg, data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv"])
    print("xong:", out, flush=True)


if __name__ == "__main__":
    main()
