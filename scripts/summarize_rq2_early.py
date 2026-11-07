"""Bảng 3 của Review 07/11 truy vết được (Bảng 7 bước 7): đọc zip thử nghiệm sớm, không chạy lại mô hình.

Tính từ rq2_mvp_per_run.csv, rq2_mvp_gamma_path.csv và rq2_early_select_on_source.csv: AUROC trung bình theo n_T × k của target-only, m5_cv (M5 đã đăng ký),
m5_pos (QĐ49), từng γ cố định và select-on-source, hàng "all" là trung bình các ô; so cặp theo lần chạy; phân bố γ được chọn.
Đây là dữ liệu phát triển: không trích như kết quả (QĐ54).

    python scripts/summarize_rq2_early.py --zip rq2_early_results.zip

Đầu ra: results/survey_v2/RQ2_mvp_early/rules/{rq2_early_rules_summary.csv, rq2_early_rules_pairs.csv, rq2_early_rules_gamma.csv, config.json}.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2.data import sha256  # noqa: E402
from survey_v2.mvp import pick_gamma, pick_gamma_positive  # noqa: E402
from survey_v2.runlog import all_versions, results_dir, start_run, write_config  # noqa: E402

KEY = ["split", "subset", "n_t", "k"]
MEMBERS = ("rq2_mvp_per_run.csv", "rq2_mvp_gamma_path.csv", "rq2_early_select_on_source.csv")
RULES = ("m5_cv", "m5_pos")
TIE_TOLERANCE = 1e-12          # hai AUROC thực chất bằng nhau có thể lệch 1e-16 do số thực


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, required=True, help="zip kết quả của việc rq2_early")
    ap.add_argument("--out", default="RQ2_mvp_early/rules")
    return ap.parse_args(argv)


def load_pilot(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()

        def member(suffix):
            hits = [n for n in names if n.endswith("/" + suffix) or n == suffix]
            if len(hits) != 1:
                raise ValueError(f"{path}: cần đúng một {suffix}, thấy {len(hits)}")
            return pd.read_csv(io.BytesIO(zf.read(hits[0])), dtype={"n_t": str}, float_precision="round_trip")

        return tuple(member(m) for m in MEMBERS)


def _gamma_col(gamma: float) -> str:
    return f"gamma_{gamma:g}"


def run_table(per_run: pd.DataFrame, path: pd.DataFrame, select_on_source: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Một dòng cho mỗi (split, subset, n_t, k): AUROC của từng nhánh và γ của từng quy tắc. Trả về (bảng, các γ > 0)."""
    fixed = per_run[per_run.gamma_rule == "fixed"]
    if "prefilter_space" in fixed.columns:                       # bản sau QĐ51: γ = 0 có hai không gian, target-only là "target"
        fixed = fixed[(fixed.gamma > 0) | (fixed.prefilter_space == "target")]
    gammas = sorted(float(g) for g in fixed.gamma.unique())
    table = fixed.pivot(index=KEY, columns="gamma", values="auroc")
    table.columns = [_gamma_col(g) for g in table.columns]
    table = table.rename(columns={_gamma_col(0.0): "target_only"})
    table["select_on_source"] = select_on_source.set_index(KEY).auroc
    recorded = per_run[per_run.gamma_rule == "m5_cv"].set_index(KEY).gamma
    for key, group in path.groupby(KEY):
        logloss = dict(zip(group.gamma.astype(float), group.oof_logloss))
        cv, pos = pick_gamma(logloss), pick_gamma_positive(logloss)
        if float(recorded.loc[key]) != cv:
            raise ValueError(f"{key}: m5_cv tính lại = {cv}, trong tệp = {recorded.loc[key]}")
        for rule, gamma in (("m5_cv", cv), ("m5_pos", pos)):
            table.loc[key, f"{rule}_gamma"] = gamma
            table.loc[key, rule] = table.loc[key, "target_only" if gamma == 0 else _gamma_col(gamma)]
    return table, [g for g in gammas if g > 0]


def _pair(name: str, a: pd.Series, b: pd.Series) -> dict:
    d = a - b
    return {"comparison": name, "mean_diff": float(d.mean()), "wins": int((d > TIE_TOLERANCE).sum()), "losses": int((d < -TIE_TOLERANCE).sum()),
            "ties": int((d.abs() <= TIE_TOLERANCE).sum()), "n_pairs": len(d)}


def summarize(per_run: pd.DataFrame, path: pd.DataFrame, select_on_source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(bảng theo ô kèm hàng trung bình các ô, so cặp theo lần chạy, phân bố γ được chọn)."""
    table, positive = run_table(per_run, path, select_on_source)
    arms = ["target_only", *RULES, *map(_gamma_col, positive), "select_on_source"]
    cells = table.groupby(["n_t", "k"])[arms].mean()
    cells = cells.sort_index(level=["n_t", "k"], key=lambda s: s.astype(int)).reset_index()
    mean_row = {"n_t": "all", "k": "all", **cells[arms].mean().to_dict()}
    summary = pd.concat([cells.astype({"k": object}), pd.DataFrame([mean_row])], ignore_index=True)

    pairs = []
    for rule in RULES:
        pairs += [_pair(f"{rule} - {other}", table[rule], table[other]) for other in ("target_only", "select_on_source", *map(_gamma_col, positive))]
    pairs += [_pair(f"{_gamma_col(g)} - select_on_source", table[_gamma_col(g)], table["select_on_source"]) for g in positive]
    first = _gamma_col(positive[0])
    off = table[table.m5_cv_gamma == 0]
    pairs.append(_pair(f"m5_cv chose gamma=0: m5_cv - {first}", off.m5_cv, off[first]))

    gammas = [{"rule": rule, "gamma": float(g), "n_runs": int(n)} for rule in RULES for g, n in table[f"{rule}_gamma"].value_counts().sort_index().items()]
    return summary, pd.DataFrame(pairs), pd.DataFrame(gammas)


def main(argv=None, script=__file__) -> int:
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t0 = time.time()
    summary, pairs, gammas = summarize(*load_pilot(args.zip))
    out = results_dir(args.out)
    for name, frame in (("rq2_early_rules_summary", summary), ("rq2_early_rules_pairs", pairs), ("rq2_early_rules_gamma", gammas)):
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    cfg = {"experiment": "RQ2_mvp_early_rules", "input_zip": str(args.zip), "input_zip_sha256": sha256(args.zip),
           "note": "dữ liệu phát triển; không trích như kết quả (QĐ54)", "elapsed_s": time.time() - t0, "versions": all_versions()}
    write_config(out, prov, cfg, data_files=[args.zip])
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.round(4).to_string(index=False), pairs.round(4).to_string(index=False), gammas.to_string(index=False), sep="\n\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
