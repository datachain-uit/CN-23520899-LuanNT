"""scripts/build_gse30219_bas.py (Review 08/11 Mục 8.5 bước 1–2, QĐ53b, QĐ67 N4): biến thể BAS → LUSC của GSE30219.

Dữ liệu giả: series matrix và annotation GPL570 nhỏ, bản R5 dựng từ chính series matrix đó bằng cách lấy probe phương sai lớn nhất.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load():
    spec = importlib.util.spec_from_file_location("build_gse30219_bas_under_test", REPO / "scripts" / "build_gse30219_bas.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load()

GENES = ["XIST", "RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY"] + [f"G{i}" for i in range(30)]


def _fixture(tmp_path: Path, n_adc=14, n_sqc=12, n_bas=8, dup=None, flip_sex=None, seed=0):
    """Series matrix (2 probe mỗi gen, thêm probe không có symbol), annotation, thư mục R5 v3 (đã bỏ 1 mẫu QC).

    dup = (bas_index, r5_index): mẫu BAS là bản sao của một mẫu R5. flip_sex = chỉ số mẫu BAS ghi giới tính ngược biểu hiện."""
    rng = np.random.default_rng(seed)
    hist = ["ADC"] * n_adc + ["SQC"] * n_sqc + ["BAS"] * n_bas + ["LCNE"] * 3
    n = len(hist)
    ids = [f"GSM{1000 + i}" for i in range(n)]
    sex = np.array(["M" if i % 3 else "F" for i in range(n)])
    probes, symbols, rows = [], [], []
    for g in GENES:
        for rep in range(2):
            base = rng.normal(6, 1.0 if rep == 0 else 0.3, size=n)      # probe 0 có phương sai lớn hơn
            if g == "XIST":
                base = np.where(sex == "F", 11.0, 4.0) + rng.normal(0, 0.2, n)
            elif g in ("RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY"):
                base = np.where(sex == "M", 10.0, 3.0) + rng.normal(0, 0.2, n)
            probes.append(f"{g}_p{rep}"); symbols.append(f"{g} /// OTHER" if rep == 1 and g == "G0" else g)
            rows.append(np.round(base, 4))
    probes.append("CTRL_p"); symbols.append(""); rows.append(rng.normal(size=n))
    E = np.vstack(rows)
    if dup is not None:
        b, r = dup
        E[:, n_adc + n_sqc + b] = E[:, r]
    meta_sex = sex.copy()
    if flip_sex is not None:
        k = n_adc + n_sqc + flip_sex
        meta_sex[k] = "F" if sex[k] == "M" else "M"
    sm = tmp_path / "GSE_series_matrix.txt.gz"
    with gzip.open(sm, "wt", encoding="utf-8") as h:
        h.write("!Series_title\t\"toy\"\n")
        h.write("!Sample_title\t" + "\t".join(f'"tumor {i}"' for i in range(n)) + "\n")
        h.write("!Sample_geo_accession\t" + "\t".join(f'"{s}"' for s in ids) + "\n")
        h.write("!Sample_platform_id\t" + "\t".join('"GPL570"' for _ in ids) + "\n")
        h.write("!Sample_characteristics_ch1\t" + "\t".join(f'"histology: {x}"' for x in hist) + "\n")
        h.write("!Sample_characteristics_ch1\t" + "\t".join(f'"gender: {x}"' for x in meta_sex) + "\n")
        h.write("!series_matrix_table_begin\n")
        h.write("\"ID_REF\"\t" + "\t".join(f'"{s}"' for s in ids) + "\n")
        for p, row in zip(probes, E):
            h.write(f'"{p}"\t' + "\t".join(f"{v:.4f}" for v in row) + "\n")
        h.write("!series_matrix_table_end\n")
    ann = tmp_path / "GPL570.annot.gz"
    with gzip.open(ann, "wt", encoding="utf-8") as h:
        h.write("^Annotation\n!platform_table_begin\nID\tGene symbol\n")
        for p, s in zip(probes, symbols):
            h.write(f"{p}\t{s}\n")
        h.write("!platform_table_end\n")
    # bản R5 v3: ADC + SQC, probe phương sai lớn nhất (như collapse_probes), bỏ mẫu đầu tiên theo QC
    expr, meta = m.parse_series_matrix(sm)
    r5 = [s for s, x in zip(ids, hist) if x in ("ADC", "SQC")]
    chosen = {g: max((f"{g}_p0", f"{g}_p1"), key=lambda p: expr.loc[p, r5].var()) for g in GENES}
    X = pd.DataFrame({g: expr.loc[chosen[g], r5].to_numpy(np.float32) for g in sorted(GENES)}, index=pd.Index(r5, name="sample_id"))
    base = tmp_path / "gse30219"
    base.mkdir()
    excluded = [r5[0]]
    X.drop(index=excluded).to_hdf(base / "expression.h5", key="expression", mode="w")
    md = meta.loc[r5].copy()
    md["label"] = [0 if x == "ADC" else 1 for x in np.array(hist)[:len(r5)]]
    md["sex"] = [{"M": "male", "F": "female"}[x] for x in meta_sex[:len(r5)]]
    md.drop(index=excluded).to_csv(base / "metadata.csv", index_label="sample_id")
    pd.DataFrame({"sample_id": excluded, "reason": ["QĐ22 sex check: inconsistent"], "decision": ["QĐ22"],
                  "verdict": ["inconsistent"], "category": ["both-high"]}).to_csv(base / "excluded_samples_gse30219.csv", index=False)
    return sm, ann, base, ids, hist


def test_probe_map_recovers_the_probes_of_the_reference_and_values_are_byte_identical(tmp_path):
    sm, ann, base, ids, hist = _fixture(tmp_path)
    expr, _ = m.parse_series_matrix(sm)
    ref = pd.read_hdf(base / "expression.h5", key="expression")
    pmap = m.probe_map(expr, ref, m.load_symbols(ann))
    assert set(pmap) == set(ref.columns) and all(pmap[g].startswith(g + "_") for g in pmap)
    variant = m.build_variant(sm, ann, base)
    X = variant.X.loc[ref.index, ref.columns]
    assert np.array_equal(X.to_numpy(np.float32), ref.to_numpy(np.float32))


def test_probe_map_refuses_when_reference_does_not_match_any_probe(tmp_path):
    sm, ann, base, *_ = _fixture(tmp_path)
    expr, _ = m.parse_series_matrix(sm)
    ref = pd.read_hdf(base / "expression.h5", key="expression")
    ref.iloc[0, 3] += 0.5
    with pytest.raises(ValueError, match="không khớp"):
        m.probe_map(expr, ref, m.load_symbols(ann))


def test_variant_labels_bas_as_lusc_and_keeps_only_adc_sqc_bas(tmp_path):
    sm, ann, base, ids, hist = _fixture(tmp_path)
    v = m.build_variant(sm, ann, base)
    meta = v.meta
    assert dict(meta.histology_raw.value_counts()) == {"ADC": 14, "SQC": 12, "BAS": 8}
    assert (meta.loc[meta.histology_raw == "BAS", "label"] == 1).all() and (meta.loc[meta.histology_raw == "BAS", "class"] == "LUSC").all()
    assert set(meta.loc[meta.histology_raw != "BAS", "variant_group"]) == {"r5"} and (meta.patient_id == meta.index).all()
    assert list(meta.index[:25]) == list(pd.read_hdf(base / "expression.h5", key="expression").index)    # 141 mẫu R5 giữ thứ tự cũ, đứng đầu


def _decisions(tmp_path, rows):
    path = tmp_path / "duplicate_decisions.csv"
    pd.DataFrame(rows, columns=["sample_a", "sample_b", "decision", "reason"]).to_csv(path, index=False)
    return m.load_decisions(path)


def test_qc_inherits_r5_exclusions_and_quarantines_sex_mismatch_and_applies_pair_decision(tmp_path):
    sm, ann, base, ids, hist = _fixture(tmp_path, dup=(2, 5), flip_sex=4)
    v = m.build_variant(sm, ann, base)
    bas_ids = list(v.meta.index[v.meta.histology_raw == "BAS"])
    dec = _decisions(tmp_path, [(ids[5], bas_ids[2], "drop_b", "bản sao kỹ thuật của mẫu R5; giữ mẫu R5")])   # thứ tự cặp đảo so với bảng sàng
    excluded, sex_tab, pairs = m.run_qc(v, dec)
    reasons = excluded.set_index("sample_id").reason
    r5_first = pd.read_csv(base / "excluded_samples_gse30219.csv").sample_id.iloc[0]
    assert "kế thừa" in reasons[r5_first]
    assert "opposite" in reasons[bas_ids[4]]
    assert bas_ids[2] in reasons.index and "drop_b" in reasons[bas_ids[2]] and "r_z" in reasons[bas_ids[2]]
    assert ids[5] not in reasons.index                                   # mẫu còn lại của cặp được giữ theo quyết định
    assert len(pairs) >= 1 and set(pairs.iloc[0][["sample_a", "sample_b"]]) == {bas_ids[2], ids[5]} and pairs.iloc[0].decision == "drop_b"


def test_keep_both_keeps_both_samples(tmp_path):
    sm, ann, base, ids, hist = _fixture(tmp_path, dup=(2, 5))
    v = m.build_variant(sm, ann, base)
    bas_ids = list(v.meta.index[v.meta.histology_raw == "BAS"])
    excluded, _, pairs = m.run_qc(v, _decisions(tmp_path, [(bas_ids[2], ids[5], "keep_both", "hai bệnh nhân khác nhau (đã kiểm metadata)")]))
    assert not {bas_ids[2], ids[5]} & set(excluded.sample_id) and (pairs.decision == "keep_both").any()


def test_undecided_pair_stops_with_exit_code_2_and_writes_nothing(tmp_path, capsys):
    sm, ann, base, ids, hist = _fixture(tmp_path, dup=(2, 5))
    v = m.build_variant(sm, ann, base)
    with pytest.raises(m.UndecidedPairs) as err:
        m.run_qc(v, m.load_decisions(None))
    assert len(err.value.pairs) >= 1
    out = tmp_path / "gse30219_bas"
    assert m.build_and_write(sm, ann, base, out, prov={}, decisions_path=None) == 2
    assert not out.exists() and "chưa có quyết định" in capsys.readouterr().out


def test_decision_file_is_validated(tmp_path):
    with pytest.raises(ValueError, match="decision"):
        _decisions(tmp_path, [("GSM1", "GSM2", "drop_both", "x")])
    with pytest.raises(ValueError, match="lý do"):
        _decisions(tmp_path, [("GSM1", "GSM2", "keep_both", "")])
    with pytest.raises(ValueError, match="trùng"):
        _decisions(tmp_path, [("GSM1", "GSM2", "keep_both", "a"), ("GSM2", "GSM1", "drop_a", "b")])


def test_committed_decision_file_loads():
    dec = m.load_decisions(m.DECISIONS_DEFAULT)
    assert list(dec.columns)[:4] == ["sample_a", "sample_b", "decision", "reason"]


def test_write_outputs_layout_counts_and_refuses_overwrite(tmp_path):
    sm, ann, base, ids, hist = _fixture(tmp_path)
    out = tmp_path / "gse30219_bas"
    cfg = m.write_variant(m.build_variant(sm, ann, base), out, prov={"traceable": True})
    X = pd.read_hdf(out / "expression.h5", key="expression")
    meta = pd.read_csv(out / "metadata.csv", index_col="sample_id")
    excl = pd.read_csv(out / "excluded_samples_gse30219_bas.csv")
    assert list(X.index) == list(meta.index) and not set(excl.sample_id) & set(X.index)
    assert cfg["n_per_class"] == [int((meta.label == 0).sum()), int((meta.label == 1).sum())]
    assert cfg["n_bas_kept"] == int((meta.histology_raw == "BAS").sum()) and 0 < cfg["bas_fraction_of_lusc"] < 1
    assert cfg["r5_values_identical"] is True and json.loads((out / "config.json").read_text(encoding="utf-8"))["traceable"] is True
    assert (out / "qc" / "sex_check_gse30219_bas.csv").exists() and (out / "qc" / "rz_screen_pairs_gse30219_bas.csv").exists()
    with pytest.raises(FileExistsError):
        m.write_variant(m.build_variant(sm, ann, base), out, prov={})
