"""Đầu vào cho baseline R của RQ2 (QĐ68, Review 08/11 Mục 8.3 bước 1–2): ghi ra CSV những gì scripts/r/rq2_r_baselines.R cần.

Cho mỗi cohort đích (cùng cách dựng với scripts/run_rq2_mvp.py nên cùng gen, cùng lần chạy, cùng fold với MVP):
  * không gian gen G = gen chung của đích và TCGA (survey_v2.transfer.common_genes), lọc 5.000 gen phương sai lớn nhất trên nguồn
    (variance_prefilter, = TransferContext.pre của MVP). Các cohort có cùng danh sách 5.000 gen dùng chung một thư mục space_<mã>;
  * nguồn: z-score theo thống kê của nguồn (M7, ddof = 0), nhãn, foldid 10 fold phân tầng cố định cho cv.glmnet giai đoạn 1;
  * đích: biểu hiện log2 GỐC trên 5.000 gen (R z-score theo L_T của từng lần chạy, M7), nhãn;
  * từng lần chạy của fewshot_splits.csv: mẫu train/test, foldid trong L_T = lần lặp đầu của inner_splits với
    seed_for(cid, seed, subset, n_t, 9) (đúng fold của đường γ của MVP), seed R = seed_for(cid, seed, subset, n_t, 13).

    python scripts/export_rq2_inputs.py                 # $KLTN_DATA_ROOT/results/survey_v2/RQ2_r_inputs

Đầu ra: space_<mã>/{genes.csv, source_z.csv.gz, source_y.csv, source_foldid.csv}; <cohort>/{space.txt, target_x.csv.gz, target_y.csv,
runs.csv, membership.csv}; config.json.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import align_to_genes, common_genes, load_dataset, variance_prefilter  # noqa: E402
from survey_v2.fewshot import fewshot_runs, validate_fewshot_manifest  # noqa: E402
from survey_v2.methods import zscore_apply, zscore_fit  # noqa: E402
from survey_v2.mvp import INNER_FOLDS, inner_splits  # noqa: E402
from survey_v2.runlog import all_versions, results_dir, start_run, write_config  # noqa: E402
from scripts.build_fewshot_splits import fewshot_path  # noqa: E402

GAMMA_SEED_TAG = 9            # = run_rq2_mvp.GAMMA_SEED_TAG: fold trong L_T của đường γ
R_SEED_TAG = 13               # seed R của từng lần chạy (glmtrans)
STAGE1_FOLDS = 10
STAGE1_SEED = C.seed_for(0, 0, 0, 11)     # foldid của cv.glmnet giai đoạn 1, chung cho mọi không gian gen
FLOAT_FORMAT = "%.8g"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.FEWSHOT_COHORTS))
    ap.add_argument("--source", default=C.SOURCE_DATASET)
    ap.add_argument("--prefilter", type=int, default=C.PREFILTER_P0)
    ap.add_argument("--n-t", nargs="+", default=None, help="chỉ các mức n_T này (mặc định mọi mức trong bản chia)")
    ap.add_argument("--splits-root", type=Path, default=None)
    ap.add_argument("--out", default="RQ2_r_inputs")
    return ap.parse_args(argv)


def space_id(genes) -> str:
    return hashlib.sha256("\n".join(genes).encode("utf-8")).hexdigest()[:12]


def stage1_foldid(y: np.ndarray, n_folds: int = STAGE1_FOLDS, seed: int = STAGE1_SEED) -> np.ndarray:
    fold = np.zeros(len(y), dtype=int)
    for f, (_, va) in enumerate(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y), start=1):
        fold[va] = f
    return fold


def lt_foldid(y_train: np.ndarray, seed: int) -> np.ndarray:
    """foldid 1..F của lần lặp ĐẦU của inner_splits (như đường γ của MVP), theo thứ tự mẫu train."""
    splits = inner_splits(y_train, seed)
    n_folds = min(INNER_FOLDS, int(min(np.sum(y_train == 0), np.sum(y_train == 1))))
    fold = np.zeros(len(y_train), dtype=int)
    for f, (_, va) in enumerate(splits[:n_folds], start=1):
        fold[va] = f
    if (fold == 0).any():
        raise ValueError("foldid thiếu mẫu")
    return fold


def export_cohort(source, ds, prefilter: int, splits_path: Path, n_t_filter, out: Path, written_spaces: dict) -> dict:
    genes = common_genes(ds, source)
    Xs = align_to_genes(source, genes)
    pre = variance_prefilter(Xs, prefilter)
    pre_genes = [str(g) for g in np.asarray(genes)[pre]]
    sid = space_id(pre_genes)
    sdir = out / f"space_{sid}"
    if sid not in written_spaces:
        sdir.mkdir(parents=True, exist_ok=True)
        mu, sd = zscore_fit(Xs[:, pre])
        pd.DataFrame({"order": np.arange(1, len(pre_genes) + 1), "gene": pre_genes}).to_csv(sdir / "genes.csv", index=False)
        pd.DataFrame(zscore_apply(Xs[:, pre], mu, sd), columns=pre_genes).to_csv(sdir / "source_z.csv.gz", index=False, float_format=FLOAT_FORMAT)
        pd.DataFrame({"sample_id": source.sample_ids, "y": source.y.astype(int)}).to_csv(sdir / "source_y.csv", index=False)
        pd.DataFrame({"foldid": stage1_foldid(source.y)}).to_csv(sdir / "source_foldid.csv", index=False)
        written_spaces[sid] = {"n_source": int(source.n), "n_genes": len(pre_genes), "gene_space_size": int(len(genes)), "cohorts": []}
    written_spaces[sid]["cohorts"].append(ds.name)
    cdir = out / ds.name
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "space.txt").write_text(f"space_{sid}\n", encoding="utf-8")
    Xt = align_to_genes(ds, genes)[:, pre]
    pd.DataFrame(Xt, index=pd.Index(ds.sample_ids, name="sample_id"), columns=pre_genes).to_csv(cdir / "target_x.csv.gz", float_format=FLOAT_FORMAT)
    pd.DataFrame({"sample_id": ds.sample_ids, "y": ds.y.astype(int)}).to_csv(cdir / "target_y.csv", index=False)
    man = pd.read_csv(splits_path, dtype={"n_t": str})
    errors = validate_fewshot_manifest(man)
    if errors:
        raise ValueError(f"{ds.name}: fewshot_splits.csv không hợp lệ: {errors}")
    cid = C.cohort_id(ds.name)
    y_of = dict(zip(ds.sample_ids, ds.y.astype(int)))
    runs, members = [], []
    for split, seed, subset, n_t, train_ids, test_ids in fewshot_runs(man):
        if n_t_filter and n_t not in n_t_filter:
            continue
        n_t_int = 10**6 if n_t == "pool" else int(n_t)
        run_id = f"{split}_{subset}_{n_t}"
        y_tr = np.array([y_of[s] for s in train_ids])
        fold = lt_foldid(y_tr, C.seed_for(cid, seed, subset, n_t_int, GAMMA_SEED_TAG))
        runs.append({"run_id": run_id, "split": split, "seed": seed, "subset": subset, "n_t": n_t, "n_train": len(train_ids), "n_test": len(test_ids),
                     "n_folds": int(fold.max()), "r_seed": C.seed_for(cid, seed, subset, n_t_int, R_SEED_TAG)})
        members += [{"run_id": run_id, "sample_id": s, "role": "train", "foldid": int(f)} for s, f in zip(train_ids, fold)]
        members += [{"run_id": run_id, "sample_id": s, "role": "test", "foldid": 0} for s in test_ids]
    pd.DataFrame(runs).to_csv(cdir / "runs.csv", index=False)
    pd.DataFrame(members).to_csv(cdir / "membership.csv", index=False)
    return {"space": f"space_{sid}", "n_runs": len(runs), "n_samples": int(ds.n), "splits_file": str(splits_path)}


def main(argv=None, script=__file__) -> int:
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t0 = time.time()
    from survey_v2.runlog import DATA_ROOT
    source = load_dataset(DATA_ROOT, args.source, gene_space=None)
    out = results_dir(args.out)
    spaces, cohorts = {}, {}
    for name in args.datasets:
        ds = load_dataset(DATA_ROOT, name)
        path = fewshot_path(ds, args.splits_root) if args.splits_root else ds.data_dir / "fewshot_splits.csv"
        cohorts[name] = export_cohort(source, ds, args.prefilter, path, args.n_t, out, spaces)
        print(f"[{name}] {cohorts[name]}", flush=True)
    cfg = {"experiment": "RQ2_r_inputs", "source": args.source, "prefilter": args.prefilter, "spaces": spaces, "cohorts": cohorts,
           "stage1_folds": STAGE1_FOLDS, "stage1_seed": STAGE1_SEED, "lt_fold_rule": f"inner_splits lần lặp đầu, seed_for(cid, seed, subset, n_t, {GAMMA_SEED_TAG})",
           "r_seed_rule": f"seed_for(cid, seed, subset, n_t, {R_SEED_TAG})", "source_scaling": "z-score theo nguồn (ddof = 0)",
           "target_values": "log2 gốc; R z-score theo L_T (ddof = 0, sd = 0 -> 1)", "float_format": FLOAT_FORMAT,
           "elapsed_s": time.time() - t0, "versions": all_versions()}
    write_config(out, prov, cfg)
    print("ghi", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
