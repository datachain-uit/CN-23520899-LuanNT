"""Code J — kiểm tra chéo nhãn LUAD/LUSC của 4 cohort GEO với series matrix gốc (Review 24/10; G-04, PB-02, PB-03).

Với mỗi cohort:
1. Đọc series matrix gốc (``!Sample_geo_accession``, ``!Sample_title``, mọi dòng ``!Sample_characteristics_ch1``).
2. Lấy giá trị nhãn bằng KHOÁ của trường (phần trước dấu ':'), không theo vị trí cột, không regex.
3. Ánh xạ nhãn bằng khớp CHÍNH XÁC chuỗi "khoá: giá trị" theo ``label_mapping.json`` → LUAD = 0, LUSC = 1;
   giá trị khác (large, NTL, ...) → không thuộc bài toán.
4. Trừ các mẫu trong ``excluded_samples_<cohort>.csv`` (QĐ21, QĐ22, QĐ22b).
5. So với ``metadata.csv``: tập mẫu trùng hoàn toàn, nhãn trùng 100 %, số mẫu mỗi lớp = DATASET_EXPECTED_N.

Đầu ra (trong --out): ``label_crosscheck_<cohort>.csv`` (từng mẫu của series matrix) và ``label_crosscheck_summary.csv``.
Mã thoát 0 khi cả 4 cohort đạt.

    python check_label_crosscheck.py --raw D:/KLTN/geo_target_work/raw \
        --data D:/KLTN/_kaggle_upload/experiment-data-v3/experiment-data-geo-target --out <data_root>/qc
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_N = {"gse10245": (37, 18), "gse30219": (81, 60), "gse37745": (104, 65), "gse42127": (126, 41)}  # survey_v2/config.py
# Khoá trường nhãn trong series matrix (phần trước ':' của giá trị label_mapping) — để đọc độc lập với cách dựng metadata.csv.
LABEL_KEY = {"gse10245": "disease state", "gse30219": "histology", "gse37745": "histology", "gse42127": "histology"}


def read_series_matrix(path: Path) -> list[dict]:
    ids, titles, chars = None, None, []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            row = next(csv.reader([line.rstrip("\n")], delimiter="\t"))
            tag, vals = row[0], row[1:]
            if tag == "!Sample_geo_accession":
                ids = vals
            elif tag == "!Sample_title":
                titles = vals
            elif tag == "!Sample_characteristics_ch1":
                chars.append(vals)
    assert ids, f"{path}: không thấy !Sample_geo_accession"
    out = []
    for j, gsm in enumerate(ids):
        fields = {}
        for row in chars:
            v = row[j].strip() if j < len(row) else ""
            if ":" in v:
                k = v.split(":", 1)[0].strip()
                assert k not in fields, f"{gsm}: khoá '{k}' xuất hiện hai lần"
                fields[k] = v
        out.append({"sample_id": gsm, "title": titles[j] if titles else "", "fields": fields})
    return out


def exact_mapping(label_mapping: dict, key: str) -> dict[str, int]:
    """Giá trị đầy đủ "khoá: giá trị" -> nhãn. label_mapping của GSE30219 ghi giá trị rút gọn (ADC/SQC) nên ghép lại với khoá."""
    m = {}
    for raw, spec in label_mapping["mapping"].items():
        full = raw if raw.startswith(key + ":") else f"{key}: {raw}"
        m[full] = int(spec["label"])
    return m


def check(cohort: str, raw_dir: Path, data_dir: Path, out_dir: Path) -> dict:
    d = data_dir / cohort
    lm = json.loads((d / "label_mapping.json").read_text(encoding="utf-8"))
    key = LABEL_KEY[cohort]
    mapping = exact_mapping(lm, key)
    sm_path = raw_dir / cohort.upper() / f"{cohort.upper()}_series_matrix.txt.gz"
    samples = read_series_matrix(sm_path)

    with open(d / f"excluded_samples_{cohort}.csv", encoding="utf-8-sig") as f:
        excluded = {r["sample_id"]: r.get("decision", "") for r in csv.DictReader(f)}
    with open(d / "metadata.csv", encoding="utf-8-sig") as f:
        meta = {r["sample_id"]: r for r in csv.DictReader(f)}

    rows, raw_counts, derived = [], Counter(), {}
    for s in samples:
        val = s["fields"].get(key, "")
        raw_counts[val or "(không có trường)"] += 1
        lab = mapping.get(val)
        status = "khong_thuoc_bai_toan" if lab is None else ("cach_ly_QC" if s["sample_id"] in excluded else "giu")
        if status == "giu":
            derived[s["sample_id"]] = lab
        m = meta.get(s["sample_id"])
        meta_lab = int(m["label"]) if m else None
        rows.append({"cohort": cohort, "sample_id": s["sample_id"], "title": s["title"], "source_value": val,
                     "label_from_series_matrix": "" if lab is None else lab, "status": status,
                     "qc_decision": excluded.get(s["sample_id"], ""),
                     "in_metadata": m is not None, "label_metadata": "" if meta_lab is None else meta_lab,
                     "match": (status == "giu" and meta_lab == lab) or (status != "giu" and m is None)})

    only_derived = sorted(set(derived) - set(meta))
    only_meta = sorted(set(meta) - set(derived))
    mismatched = sorted(s for s in set(derived) & set(meta) if int(meta[s]["label"]) != derived[s])
    excluded_not_in_raw = sorted(set(excluded) - {s["sample_id"] for s in samples})
    n = (sum(v == 0 for v in derived.values()), sum(v == 1 for v in derived.values()))
    n_pre = (sum(1 for s in samples if mapping.get(s["fields"].get(key, "")) == 0),
             sum(1 for s in samples if mapping.get(s["fields"].get(key, "")) == 1))
    n_mapping_declared = tuple(int(sp.get("n_samples", -1)) for sp in sorted(lm["mapping"].values(), key=lambda x: x["label"]))
    titles = [s["title"] for s in samples if s["sample_id"] in derived]
    ok = not only_derived and not only_meta and not mismatched and not excluded_not_in_raw and n == EXPECTED_N[cohort]

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"label_crosscheck_{cohort}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return {"cohort": cohort, "series_matrix": sm_path.name, "label_key": key,
            "n_series_matrix": len(samples), "n_luad_lusc_before_qc": f"{n_pre[0]}/{n_pre[1]}",
            "n_declared_in_label_mapping": "/".join(map(str, n_mapping_declared)),
            "n_excluded_qc": len(excluded), "n_derived_after_qc": f"{n[0]}/{n[1]}",
            "n_metadata": len(meta), "expected_n": f"{EXPECTED_N[cohort][0]}/{EXPECTED_N[cohort][1]}",
            "only_in_series_matrix": ";".join(only_derived), "only_in_metadata": ";".join(only_meta),
            "label_mismatch": ";".join(mismatched), "excluded_not_in_series_matrix": ";".join(excluded_not_in_raw),
            "duplicate_titles": len(titles) - len(set(titles)),
            "other_values": "; ".join(f"{k} ({v})" for k, v in sorted(raw_counts.items()) if k not in mapping),
            "pass": ok}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cohorts", nargs="+", default=list(EXPECTED_N))
    a = ap.parse_args()
    res = [check(c, a.raw, a.data, a.out) for c in a.cohorts]
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(a.out / "label_crosscheck_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]) + ["checked_at_utc"])
        w.writeheader(); w.writerows([{**r, "checked_at_utc": stamp} for r in res])
    for r in res:
        print(f"{r['cohort']}: {'ĐẠT' if r['pass'] else 'KHÔNG ĐẠT'} | series matrix {r['n_series_matrix']} mẫu, "
              f"LUAD/LUSC trước QC {r['n_luad_lusc_before_qc']} (label_mapping ghi {r['n_declared_in_label_mapping']}), "
              f"cách ly {r['n_excluded_qc']} → {r['n_derived_after_qc']} = kỳ vọng {r['expected_n']}; "
              f"chỉ ở SM: [{r['only_in_series_matrix']}] chỉ ở metadata: [{r['only_in_metadata']}] lệch nhãn: [{r['label_mismatch']}]")
    return 0 if all(r["pass"] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
