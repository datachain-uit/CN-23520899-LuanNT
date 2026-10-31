"""Tiến trình con đo bộ nhớ đỉnh của MỘT lần chọn gen (v3.1, Y19). Gọi bởi survey_v2.timing.measure_isolated().

    python -m survey_v2.timing_child --npz xy.npz --method f_test --k 10 --seed 0
In ra một dòng JSON: {"method", "k", "wall_s", "maxrss_mb", "peak_source", "ru_maxrss_mb"}; method = "none" chỉ nạp dữ liệu
(mức nền). v3.1.6 (QĐ27): maxrss_mb = `peak_rss_mb()` (VmHWM của chính tiến trình con trên Linux); ru_maxrss_mb chỉ để đối
chiếu — trên Linux nó kế thừa đỉnh RSS của tiến trình cha nên KHÔNG phải bộ nhớ của tiến trình con.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--generations", type=int, default=None)
    args = ap.parse_args(argv)
    from survey_v2.timing import _maxrss_mb, peak_rss_mb
    d = np.load(args.npz)
    X, y = np.ascontiguousarray(d["X"]), d["y"].astype(int)
    t0 = time.perf_counter()
    if args.method != "none":
        from threadpoolctl import threadpool_limits
        from survey_v2.methods import select
        kw = {"pop": args.pop, "generations": args.generations} if args.pop else None
        with threadpool_limits(1):
            select(args.method, X, y, [args.k], seed=args.seed, n_jobs=1, wrapper_kwargs=kw)
    print(json.dumps({"method": args.method, "k": args.k, "wall_s": time.perf_counter() - t0, "maxrss_mb": peak_rss_mb(),
                      "peak_source": "VmHWM" if os.path.exists("/proc/self/status") else "ru_maxrss",
                      "ru_maxrss_mb": _maxrss_mb()}), flush=True)


if __name__ == "__main__":
    main()
