"""Một lệnh sinh lại mọi bảng và hình RQ2, RQ4 của bài 2 từ các zip kết quả đã nộp (PG-04; lịch 22/11 của KBTN v2.6).

Không chạy lại mô hình. Gọi lần lượt, mỗi script một tiến trình (mỗi bảng có config.json và provenance riêng, cùng kiểm traceable):
  1. make_rq2_tables.py   --input <rq2_mvp> --r-input <rq2_r_all>          -> RQ2_tables/      (H2, H2b, H2c; so sánh mô tả ptLasso, Trans-GLM)
  2. make_w1_r_labels.py  --rq2-tables <RQ2_tables của bước 1>            -> RQ2_r_labels/    (nhãn đọc QĐ97e, W1)
  3. make_rq4_tables.py   --rq2 <rq2_mvp> --rq4 <rq4_h4b_noihc>            -> RQ4_tables/      (H4a, H4b, γ, ablation, bỏ IHC)
  4. make_qd97_tables.py  --rq2 <rq2_mvp> --rq4 <rq4_h4b_noihc>            -> RQ2_qd97_tables/ (phân rã QĐ97a, b, d)
  5. make_qd126_tables.py --rq2 <rq2_mvp> --rq4 <rq4_h4b_noihc>            -> RQ2_qd126_tables/
  6. make_qd182_tables.py --rq2 <rq2_mvp> --rq4 <rq4_h4b_noihc> --d3 <qd182_d3_shuffled> -> RQ2_qd182_tables/
  7. make_rq2_rq4_figures.py --tables <results/survey_v2>                 -> RQ2_RQ4_figures/ (6 hình PDF)
Rồi, nếu có --reference, đối chiếu từng CSV với zip bảng đã nộp (số: sai khác tuyệt đối ≤ 1e-12; chữ: trùng) và ghi
RQ2_RQ4_all/reference_check.csv. Một bước lỗi hoặc một CSV lệch -> mã thoát 1.

    python scripts/make_rq2_rq4_all.py --zips <thư mục chứa các zip> [--reference <thư mục chứa các zip bảng đã nộp>]

Zip đầu vào tìm theo mẫu tên trong --zips (đúng một tệp mỗi mẫu), hoặc chỉ định riêng bằng --rq2, --rq4, --r-all, --d3.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import sha256  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402

INPUT_PATTERNS = {"rq2": "rq2_mvp_*_results.zip", "rq4": "rq4_h4b_noihc_*_results.zip", "r_all": "rq2_r_all_*_results.zip",
                  "d3": "qd182_d3_shuffled_*_results.zip"}
# thư mục kết quả -> mẫu tên zip bảng đã nộp (lấy bản mới nhất theo tên commit trong danh sách ưu tiên nếu có nhiều)
REFERENCE = {"RQ2_tables": "rq2_tables_*_results.zip", "RQ2_r_labels": "rq2_r_labels_*_results.zip", "RQ4_tables": "rq4_tables_*_results.zip",
             "RQ2_qd97_tables": "rq2_qd97_tables_*_results.zip", "RQ2_qd126_tables": "rq2_qd126_tables_*_results.zip",
             "RQ2_qd182_tables": "rq2_qd182_tables_*_results.zip"}
ATOL = 1e-12


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zips", type=Path, default=None, help="thư mục chứa các zip kết quả (tìm theo mẫu tên)")
    for k in INPUT_PATTERNS:
        ap.add_argument(f"--{k.replace('_', '-')}", type=Path, default=None, help=f"zip {INPUT_PATTERNS[k]} (ghi đè --zips)")
    ap.add_argument("--reference", type=Path, default=None, help="thư mục chứa zip bảng đã nộp để đối chiếu từng CSV")
    ap.add_argument("--reference-zip", action="append", default=[], metavar="TÊN=ZIP",
                    help="chỉ định zip đối chiếu cho một thư mục, vd. RQ4_tables=rq4_tables_7eae081_results.zip (lặp được)")
    ap.add_argument("--n-boot", type=int, default=C.BOOTSTRAP_N)
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--out", default="RQ2_RQ4_all")
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------------------------- đầu vào
def find_one(folder: Path, pattern: str) -> Path:
    hit = sorted(Path(folder).glob(pattern))
    if len(hit) != 1:
        raise SystemExit(f"{folder}: cần đúng một tệp {pattern}, thấy {[p.name for p in hit]}")
    return hit[0]


def resolve_inputs(args) -> dict:
    out = {}
    for k, pat in INPUT_PATTERNS.items():
        v = getattr(args, k)
        if v is None:
            if args.zips is None:
                raise SystemExit(f"thiếu --{k.replace('_', '-')} hoặc --zips")
            v = find_one(args.zips, pat)
        if not Path(v).exists():
            raise SystemExit(f"không có {v}")
        out[k] = Path(v).resolve()
    return out


def steps(inp: dict, tables: Path, n_boot: int, figures: bool = True) -> list[tuple[str, list[str]]]:
    nb = ["--n-boot", str(n_boot)]
    s = [("make_rq2_tables.py", ["--input", str(inp["rq2"]), "--r-input", str(inp["r_all"]), *nb]),
         ("make_w1_r_labels.py", ["--rq2-tables", str(tables / "RQ2_tables")]),
         ("make_rq4_tables.py", ["--rq2", str(inp["rq2"]), "--rq4", str(inp["rq4"]), *nb]),
         ("make_qd97_tables.py", ["--rq2", str(inp["rq2"]), "--rq4", str(inp["rq4"]), *nb]),
         ("make_qd126_tables.py", ["--rq2", str(inp["rq2"]), "--rq4", str(inp["rq4"]), *nb]),
         ("make_qd182_tables.py", ["--rq2", str(inp["rq2"]), "--rq4", str(inp["rq4"]), "--d3", str(inp["d3"]), *nb])]
    if figures:
        s.append(("make_rq2_rq4_figures.py", ["--tables", str(tables)]))
    return s


# ------------------------------------------------------------------------------------------------- đối chiếu
def compare_frames(new: pd.DataFrame, ref: pd.DataFrame, atol: float = ATOL) -> tuple[bool, str]:
    """Cùng cột, cùng số dòng; cột số sai khác tuyệt đối ≤ atol (NaN trùng NaN); cột khác trùng chuỗi."""
    if list(new.columns) != list(ref.columns):
        return False, f"cột khác: {sorted(set(new.columns) ^ set(ref.columns)) or 'thứ tự'}"
    if len(new) != len(ref):
        return False, f"số dòng {len(new)} ≠ {len(ref)}"
    worst = 0.0
    for c in new.columns:
        a, b = new[c], ref[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            x, y = a.to_numpy(float), b.to_numpy(float)
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                return False, f"{c}: vị trí NaN khác"
            d = np.abs(x - y)[~np.isnan(x)]
            m = float(d.max()) if d.size else 0.0
            worst = max(worst, m)
            if m > atol:
                return False, f"{c}: sai khác lớn nhất {m:.3g}"
        elif not a.fillna("<NA>").astype(str).equals(b.fillna("<NA>").astype(str)):
            return False, f"{c}: chuỗi khác ở {int((a.fillna('<NA>').astype(str) != b.fillna('<NA>').astype(str)).sum())} dòng"
    return True, f"sai khác lớn nhất {worst:.3g}"


def _read_csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", dtype={"n_t": str, "k_avg": str})


def check_folder(folder: Path, ref_zip: Path) -> list[dict]:
    rows = []
    with zipfile.ZipFile(ref_zip) as zf:
        refs = {Path(n).name: n for n in zf.namelist() if n.endswith(".csv") and f"/{folder.name}/" in f"/{n}"}
        new_files = sorted(p.name for p in folder.glob("*.csv"))
        for name in sorted(set(refs) | set(new_files)):
            row = {"folder": folder.name, "file": name, "reference_zip": ref_zip.name}
            if name not in refs or name not in new_files:
                row.update(match=False, detail="chỉ có ở " + ("bản mới" if name in new_files else "bản đã nộp"))
            else:
                ok, detail = compare_frames(_read_csv((folder / name).read_bytes()), _read_csv(zf.read(refs[name])))
                row.update(match=ok, detail=detail)
            rows.append(row)
    return rows


def reference_zips(args) -> dict:
    chosen = {}
    for item in args.reference_zip:
        k, _, v = item.partition("=")
        chosen[k] = Path(v) if Path(v).is_absolute() or args.reference is None else args.reference / v
    if args.reference is not None:
        for k, pat in REFERENCE.items():
            if k not in chosen:
                hit = sorted(args.reference.glob(pat))
                if len(hit) > 1:
                    raise SystemExit(f"{args.reference}: nhiều zip {pat} ({[p.name for p in hit]}); chọn bằng --reference-zip {k}=<tên>")
                if hit:
                    chosen[k] = hit[0]
    return chosen


# ------------------------------------------------------------------------------------------------- chạy
def main(argv=None, script=__file__) -> int:
    for stream in (sys.stdout, sys.stderr):           # chữ tiếng Việt trên console Windows (cp1252) khi ghi ra tệp
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    inp = resolve_inputs(args)
    refs = reference_zips(args)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t0 = time.time()
    tables = DATA_ROOT.joinpath(*C.RESULT_DIR)
    log = []
    for name, extra in steps(inp, tables, args.n_boot, figures=not args.skip_figures):
        cmd = [sys.executable, str(REPO / "scripts" / name), *extra]
        print(">>", " ".join(cmd), flush=True)
        t = time.time()
        rc = subprocess.run(cmd, env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
        log.append({"script": name, "args": extra, "returncode": rc, "elapsed_s": round(time.time() - t, 1)})
        if rc != 0:
            print(f"DỪNG: {name} trả mã {rc}", file=sys.stderr)
            break
    ok_steps = all(x["returncode"] == 0 for x in log) and len(log) == len(steps(inp, tables, args.n_boot, not args.skip_figures))
    checks = []
    if ok_steps:
        for folder, z in refs.items():
            checks += check_folder(tables / folder, z)
    out = results_dir(args.out)
    chk = pd.DataFrame(checks, columns=["folder", "file", "reference_zip", "match", "detail"])
    chk.to_csv(out / "reference_check.csv", index=False, encoding="utf-8-sig")
    configs = {}
    for d in sorted(p for p in tables.iterdir() if p.is_dir() and (p / "config.json").exists()):
        c = json.loads((d / "config.json").read_text(encoding="utf-8"))
        configs[d.name] = {"commit_hash": c.get("commit_hash"), "traceable": c.get("traceable")}
    cfg = {"experiment": "RQ2_RQ4_all", "inputs": {k: str(v) for k, v in inp.items()}, "input_sha256": {v.name: sha256(v) for v in inp.values()},
           "steps": log, "reference": {k: str(v) for k, v in refs.items()},
           "reference_sha256": {v.name: sha256(v) for v in refs.values()},
           "n_reference_files": len(chk), "n_reference_match": int(chk.match.sum()) if len(chk) else 0,
           "result_configs": configs, "elapsed_s": time.time() - t0, "versions": all_versions()}
    write_config(out, prov, cfg)
    for x in log:
        print(f"{x['script']:<28} mã {x['returncode']}  {x['elapsed_s']} s")
    if len(chk):
        print(chk.groupby("folder").match.agg(["sum", "count"]).to_string())
        bad = chk[~chk.match]
        if len(bad):
            print(bad.to_string(index=False))
    good = ok_steps and (chk.match.all() if len(chk) else True)
    print("ĐẠT" if good else "KHÔNG ĐẠT", "-", out)
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
