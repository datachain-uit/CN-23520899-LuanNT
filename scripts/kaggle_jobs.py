"""Danh mục việc chạy trên Kaggle cho survey v3 (E0–E7) — một nơi duy nhất chứa lệnh, chia phiên và kiểm tra (M18).

Notebook `notebooks/kaggle_survey_v3_runner.ipynb` chỉ đặt JOB và gọi script này; mọi lệnh thí nghiệm, cách chia shard,
điều kiện đầu vào và kiểm tra đầu ra nằm ở đây (đã commit). Không có tham số giao thức nào ở đây: k, n_T, seed, lưới… lấy từ
`survey_v2/config.py` qua mặc định của từng script.

    python scripts/kaggle_jobs.py plan                     # lịch: đợt, hạn, thứ tự, đầu vào cần khôi phục, ước lượng giờ
    python scripts/kaggle_jobs.py show e1_phaseA_geo       # lệnh cụ thể của một việc
    python scripts/kaggle_jobs.py run e1_phaseA_geo        # chạy (shard song song tối đa --parallel tiến trình), rồi verify
    python scripts/kaggle_jobs.py verify e1_phaseA_geo     # config.json: traceable = true, commit_hash = HEAD
    python scripts/kaggle_jobs.py status                   # việc nào đã có đầu ra trong $KLTN_DATA_ROOT/results/survey_v2
    python scripts/kaggle_jobs.py merge-e2e3               # gộp E2_E3_p1..p3 -> E2_E3 (kiểm tra hiệu chuẩn, cùng commit)

Quy ước chia shard:
* Pha A (run_e1_feature_selection.py) không song song bên trong -> chạy nhiều tiến trình, mỗi tiến trình một cohort hoặc
  một lần lặp; checkpoint theo (repeat, fold, method) nên các tiến trình không ghi đè nhau. Bước `finalize` chạy lại lệnh
  đầy đủ một lần để gộp (mọi checkpoint đã có -> chỉ gộp).
* Pha B, E1b, E7, E4/E5 ghi nối tiếp (append) theo bộ dữ liệu -> chỉ song song khác bộ; trong một bộ dùng --n-jobs của script.
* E2/E3 đo thời gian: `exclusive` (một tiến trình, không chạy gì khác cùng phiên; BLAS 1 luồng do script tự đặt).
* Mọi script tự tiếp tục khi chạy lại -> phiên Kaggle quá 12 giờ: lưu output, gắn làm Input, chạy lại cùng JOB.

v3.1.6 (Review 31/10/2026, QĐ25–QĐ28): (1) việc `redo_svm_rfe_A` / `redo_svm_rfe_B` chọn lại và đánh giá lại SVM-RFE
(random_state) ở E1 pha A/B, ablation A, E1b, E4/E5 bằng --redo-methods, giữ nguyên mọi phương pháp khác; (2) `e2_ref`,
`e3_p1..p3`, `e2e3_merge` chạy lại ở bản này (tracemalloc và bộ nhớ tiến trình con — survey_v2/timing.py); (3) `requires`
cho e4e5/e4_tcga/e7/tables; (4) merge-e2e3 bỏ qua tệp CSV rỗng; (5) mỗi lần `run` ghi `logs/pip_freeze_<JOB>.txt`.

v3.1.7 (Review 02/11/2026, QĐ30–QĐ32): (1) `restore_from` là THỨ TỰ GIẢI NÉN trong notebook — zip sau ghi đè zip trước, nên
output mới nhất (vd. `redo_svm_rfe_A`, `redo_svm_rfe_B`, `panels`) đứng CUỐI danh sách (Phát hiện 2 của HANDOVER 04/11);
(2) `requires_cfg`: điều kiện trên config.json của đầu vào (vd. E1_feature_selection đã chọn lại svm_rfe, E2_E3 không
tracemalloc) — `run` dừng với mã 2 nếu không đạt, nên khôi phục sai thứ tự bị phát hiện trước khi chạy; (3) việc `panels`:
E1 pha A/B và E1b cho hai panel công bố (Girard 2016, Wilkerson 2013; survey_v2/panels/published_panels.csv) nối vào kết
quả đã có; (4) `tables` chạy lại sau `panels` (bảng ablation IHC và CI bootstrap tái lập — scripts/make_survey_tables.py);
(5) `status` in commit và redo_methods của từng thư mục kết quả đã khôi phục.

Bài 2 (Review 03/11/2026, QĐ37): việc `rq2_early` là thử nghiệm sớm PC-04 trên TCGA; code chỉ có ở nhánh rq2-mvp.
Review 07/11/2026: việc `rq2_mvp` chạy RQ2 trên 4 cohort; `gates` là các lệnh chạy trong bước verify, thoát khác 0 thì việc không đạt (QĐ51: cổng bằng nhau).
Review 08/11/2026 (QĐ70): `pin_versions` — trước khi chạy, numpy, pandas, scipy, scikit-learn của phiên phải trùng trường
versions trong config.json của E1b, E7 đã khôi phục; lệch thì dừng mã 2 và in lệnh pip install == cần chạy.
Review 08/11/2026 (QĐ53b, Mục 8.5): việc `rq1_sens_b` cho độ nhạy RQ1 phần (b); không cần khôi phục output nào.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402

TCGA = C.SOURCE_DATASET
GEO = tuple(C.FEWSHOT_COHORTS)
GPL570 = tuple(C.WRAPPER_DATASETS)
IHC_GENES = ("NKX2-1", "TP63", "NAPSA", "KRT5")
RQ1_BAS_DATASET = "gse30219_bas"       # biến thể độ nhạy (scripts/build_gse30219_bas.py), không thuộc C.DATASETS
RQ1_FTEST_P0 = 2000


def _published_panel_methods() -> list[str]:
    """Tên method của các panel công bố (panel_<tên>) đọc từ survey_v2/panels/published_panels.csv (v3.1.7)."""
    try:
        from survey_v2.panels import published_panels
        return sorted(published_panels())
    except Exception:  # noqa: BLE001 — thiếu tệp/gói: việc panels không có phương pháp, kiểm tra ở cmd_run
        return []


PUBLISHED_PANELS = _published_panel_methods()
E3_PARTS = {  # chia E3 theo chi phí để mỗi phiên < 12 giờ; mỗi phần tự hiệu chuẩn với phiên tham chiếu (QĐ18)
    "p1": ["f_test", "mutual_info", "mrmr", "l1_lr", "ktsp", "abess"],
    "p2": ["elastic_net", "rf_importance", "stability_selection"],
    "p3": ["relieff", "svm_rfe"],
}


@dataclass
class Job:
    name: str
    batch: str                     # đợt / hạn
    what: str
    shards: list                   # mỗi phần tử: list đối số (sau "python")
    outputs: list                  # thư mục kết quả (tương đối results/survey_v2) phải có config.json traceable
    finalize: list = field(default_factory=list)
    requires: list = field(default_factory=list)    # tệp (tương đối results/survey_v2) phải có trước khi chạy
    restore_from: list = field(default_factory=list)  # output các việc phải gắn làm Input, THEO THỨ TỰ GIẢI NÉN (mới nhất cuối)
    requires_cfg: list = field(default_factory=list)  # (config.json tương đối, khoá, giá trị): bằng nhau, hoặc thuộc danh sách
    gates: list = field(default_factory=list)       # lệnh chạy ở bước verify (mỗi phần tử: list đối số sau "python"); thoát khác 0 là không đạt
    pin_versions: list = field(default_factory=list)  # config.json tham chiếu (tương đối results/survey_v2): PINNED_PACKAGES của phiên phải trùng trường versions (QĐ70)
    parallel: int = 4
    exclusive: bool = False
    hours: str = ""


def _fs(*a):
    return ["scripts/run_e1_feature_selection.py", *a]


def _clf(*a):
    return ["scripts/run_e1_classification.py", *a]


def build_jobs() -> dict[str, Job]:
    J: list[Job] = []
    ref = "E2_E3_ref/calibration.csv"
    # ----------------------------------------------------------------- Đợt 1 (03/11): sanity, E0, E1 GEO, E2 tham chiếu
    J.append(Job("sanity", "1 · 03/11", "Sanity Y10 (normal vs tumor, AUROC >= 0,99)", [["scripts/run_sanity_auroc.py"]],
                 ["sanity"], parallel=1, hours="~1"))
    J.append(Job("e0", "1 · 03/11", "E0 kiểm tra tín hiệu, 5 bộ (200 hoán vị + 100 random-k)",
                 [["scripts/run_survey_e0_signal_check.py", "--n-jobs", "4"]], [f"E0/{d}" for d in C.DATASETS], parallel=1, hours="~1"))
    J.append(Job("e1_phaseA_geo", "1 · 03/11", "E1 pha A, 4 cohort GEO, 11 phương pháp + IHC/panel, 5×5 (một tiến trình/cohort)",
                 [_fs("--datasets", d) for d in GEO], [f"E1_feature_selection/{d}" for d in GEO], hours="3–6"))
    J.append(Job("e1_phaseB_geo", "1 · 03/11", "E1 pha B, 4 cohort GEO: LR-L2 (+RF, SVM ở k ∈ {5, 25}), B0, random-k",
                 [_clf("--datasets", d, "--secondary") for d in GEO], [f"E1_classification/{d}" for d in GEO],
                 requires=[f"E1_feature_selection/{d}/selected_genes.csv" for d in GEO], restore_from=["e1_phaseA_geo"], hours="1–2"))
    J.append(Job("e2_ref", "2 · 06/11", "E2 phiên tham chiếu = E2 chính thức (13 phương pháp trong MỘT phiên; calibration.csv; "
                 "bộ nhớ đỉnh trong tiến trình con; không E3) — CHẠY LẠI ở v3.1.6 (QĐ27: thời gian không tracemalloc, bộ nhớ VmHWM)",
                 [["scripts/run_survey_e2_e3_timing.py", "--wrappers", "--isolate-memory", "--skip-e3", "--out", "E2_E3_ref"]],
                 ["E2_E3_ref"], parallel=1, exclusive=True, hours="6–9"))
    # ----------------------------------------------------------------- Đợt 1 phần TCGA (04/11)
    J.append(Job("e1_phaseA_tcga_r012", "1T · 04/11", "E1 pha A TCGA, lần lặp 0, 1, 2 (ba tiến trình song song)",
                 [_fs("--datasets", TCGA, "--repeats", str(r)) for r in (0, 1, 2)], [], hours="5–8"))
    J.append(Job("e1_phaseA_tcga_r34", "1T · 04/11", "E1 pha A TCGA, lần lặp 3, 4 (song song với phiên r012)",
                 [_fs("--datasets", TCGA, "--repeats", str(r)) for r in (3, 4)], [], hours="5–8"))
    J.append(Job("e1_phaseB_tcga", "1T · 04/11", "Gộp pha A TCGA (5 lần lặp) rồi pha B TCGA",
                 [_clf("--datasets", TCGA, "--secondary")], [f"E1_feature_selection/{TCGA}", f"E1_classification/{TCGA}"],
                 finalize=[], parallel=1, restore_from=["e1_phaseA_tcga_r012", "e1_phaseA_tcga_r34"],
                 requires=[f"E1_feature_selection/{TCGA}/checkpoints/repeat_{r}_fold_0_f_test.genes.csv" for r in (0, 3)], hours="3–6"))
    # gộp pha A chạy TRƯỚC pha B (cùng tiến trình tuần tự)
    J[-1].shards = [_fs("--datasets", TCGA), J[-1].shards[0]]
    # ----------------------------------------------------------------- Đợt 2 (06/11): E1b, E3, E4 + E5
    J.append(Job("fewshot_e1b", "2 · 06/11", "E1b đường cong học few-shot, 4 cohort (5 split × 4 tập con)",
                 [["scripts/run_survey_e1b_learning_curve.py", "--n-jobs", "4"]], [f"E1b/{d}" for d in GEO], parallel=1, hours="3–5"))
    for part, methods in E3_PARTS.items():
        extra = ["--wrappers"] if part == "p1" else []
        J.append(Job(f"e3_{part}", "2 · 06/11", f"E3 phần {part} ({', '.join(methods)}{', GA, BPSO' if extra else ''}), hiệu chuẩn theo E2_E3_ref; "
                     "--skip-e2 (E2 chính thức = phiên tham chiếu); elastic_net đo tới p = 10.000 (QĐ23)",
                     [["scripts/run_survey_e2_e3_timing.py", "--methods", *methods, *extra, "--skip-e2",
                       "--calibration-reference", "{RESULTS}/" + ref, "--out", f"E2_E3_{part}"]],
                     [f"E2_E3_{part}"], requires=[ref], restore_from=["e2_ref"], parallel=1, exclusive=True,
                     hours={"p1": "2–3", "p2": "5–7", "p3": "7–9"}[part]))
    J.append(Job("e2e3_merge", "2 · 06/11", "Gộp E2 của phiên tham chiếu + E3 của p1..p3 -> E2_E3 (dùng cho make_survey_tables.py)",
                 [["scripts/kaggle_jobs.py", "merge-e2e3"]], ["E2_E3"],
                 requires=["E2_E3_ref/config.json"] + [f"E2_E3_{p}/config.json" for p in E3_PARTS],
                 restore_from=["e2_ref"] + [f"e3_{p}" for p in E3_PARTS], parallel=1, hours="<0,1"))
    J.append(Job("e4e5_gse37745", "2 · 06/11", "E4 (M = 30) + E5a + E5b + E5c trên GSE37745 (+GA, BPSO ở k = 10); "
                 "chạy lại sau wrap_gse37745 để E5c có GA/BPSO (phần đã có được bỏ qua)",
                 [["scripts/run_survey_e4_e5_stability.py", "--datasets", "gse37745", "--wrappers", "--n-jobs", "4"]],
                 ["E4_E5/gse37745"], parallel=1, restore_from=["e1_phaseA_geo", "wrap_gse37745"],
                 requires=["E1_feature_selection/gse37745/selected_genes.csv", "E1_wrapper_feature_selection/gse37745/selected_genes.csv"],
                 hours="4–8"))
    J.append(Job("e4_tcga", "2 · 06/11", "E4 (M = 30) + E5c trên TCGA (+GA, BPSO ở k = 10); quá 12 giờ thì chạy lại, tự tiếp tục",
                 [["scripts/run_survey_e4_e5_stability.py", "--datasets", TCGA, "--wrappers", "--skip-e5a", "--skip-e5b", "--n-jobs", "4"]],
                 [f"E4_E5/{TCGA}"], parallel=1, restore_from=["e1_phaseB_tcga"],
                 requires=[f"E1_feature_selection/{TCGA}/selected_genes.csv"], hours="6–11"))
    # ----------------------------------------------------------------- Đợt 3 (08/11): wrapper, E7
    for d in GPL570:
        J.append(Job(f"wrap_{d}", "3 · 08/11", f"E1 pha A wrapper GA + BPSO, {d}, lần lặp 0–2, k ∈ {{5, 10, 50}}",
                     [["scripts/run_e1_wrappers_all_cohorts.py", "--datasets", d, "--repeats", str(r)] for r in range(C.WRAPPER_REPEATS)],
                     [f"E1_wrapper_feature_selection/{d}"], finalize=[["scripts/run_e1_wrappers_all_cohorts.py", "--datasets", d]],
                     parallel=3, hours="3–6"))
    J.append(Job("e1_phaseB_wrappers", "3 · 08/11", "E1 pha B cho panel GA, BPSO (3 cohort GPL570), nối vào E1_classification",
                 [_clf("--datasets", d, "--methods", "ga", "bpso", "--secondary", "--no-references") for d in GPL570],
                 [f"E1_classification/{d}" for d in GPL570],
                 requires=[f"E1_wrapper_feature_selection/{d}/selected_genes.csv" for d in GPL570]
                 + [f"E1_classification/{d}/classification_per_fold.csv" for d in GPL570],
                 restore_from=["e1_phaseB_geo"] + [f"wrap_{d}" for d in GPL570], parallel=3, hours="~1"))
    J.append(Job("e7", "3 · 08/11", "E7 chuyển giao đơn giản (4 chiến lược × 5 k × 20 lần chạy × n_T ∈ {5, 10, 20})",
                 [["scripts/run_survey_e7_transfer_naive.py", "--n-jobs", "4"]], [f"E7/{d}" for d in GEO], parallel=1, hours="2–4"))
    J.append(Job("tables", "sau mỗi đợt", "Bảng, hình, provenance_manifest (E6 ghép tại k = 10; v3.1.7: + bảng ablation IHC, "
                 "panel công bố, CI bootstrap tái lập)",
                 [["scripts/make_survey_tables.py"]], ["tables"], parallel=1,
                 requires=[f"E1_classification/{d}/classification_per_fold.csv" for d in C.DATASETS] + [f"E0/{d}/e0_signal_check.csv" for d in C.DATASETS]
                 + [f"E1_classification_noIHC/{d}/classification_per_fold.csv" for d in C.DATASETS] + ["E2_E3/e2_cost_point.csv"],
                 # v3.1.7a: sau `panels`, E1_feature_selection/*/config.json ghi redo_methods = hai panel (không còn svm_rfe);
                 # việc `panels` đã đòi redo_methods ∋ svm_rfe ở đầu vào của nó, nên redo_methods ∋ panel_* bảo đảm cả chuỗi.
                 requires_cfg=[(f"E1_feature_selection/{d}/config.json", "redo_methods", m) for d in C.DATASETS for m in PUBLISHED_PANELS]
                 + [(f"E1_classification/{d}/config.json", "methods_in_results", m) for d in C.DATASETS for m in PUBLISHED_PANELS]
                 + [("E2_E3/config.json", "timing_tracemalloc", False), ("E2_E3/config.json", "peak_source", "VmHWM")],
                 restore_from=["e0", "sanity", "e7", "e2e3_merge", "ablation_phaseB", "redo_svm_rfe_B", "panels"], hours="<0,5"))
    # ----------------------------------------------------------------- v3.1.6: chọn lại SVM-RFE (QĐ25; random_state) — 02–04/11
    abl = ["--exclude-genes", *IHC_GENES, "--out", "E1_feature_selection_noIHC"]
    redo = ["--redo-methods", "svm_rfe"]
    J.append(Job("redo_svm_rfe_A", "R · 04/11", "SVM-RFE chọn lại (random_state, QĐ25): E1 pha A 5 bộ + ablation pha A 5 bộ; "
                 "phương pháp khác lấy từ checkpoint đã khôi phục",
                 [_fs("--datasets", d, *redo) for d in C.DATASETS] + [_fs("--datasets", d, *abl, *redo) for d in C.DATASETS],
                 [f"E1_feature_selection/{d}" for d in C.DATASETS] + [f"E1_feature_selection_noIHC/{d}" for d in C.DATASETS],
                 requires=[f"E1_feature_selection/{d}/checkpoints/repeat_0_fold_0_f_test.genes.csv" for d in C.DATASETS]
                 + [f"E1_feature_selection_noIHC/{d}/checkpoints/repeat_0_fold_0_f_test.genes.csv" for d in C.DATASETS],
                 restore_from=["e1_phaseA_geo", "e1_phaseB_tcga", "ablation_phaseA_geo", "ablation_phaseA_tcga"], parallel=4, hours="~1"))
    J.append(Job("redo_svm_rfe_B", "R · 04/11", "SVM-RFE đánh giá lại (QĐ25): E1 pha B 5 bộ (+RF, SVM ở k ∈ {5, 25}), E1b 4 cohort, "
                 "E4/E5 2 bộ (E5c tính lại); các dòng khác giữ nguyên",
                 [_clf("--datasets", d, "--secondary", "--no-references", "--methods", "svm_rfe", *redo) for d in C.DATASETS]
                 + [["scripts/run_survey_e1b_learning_curve.py", "--datasets", d, "--methods", "svm_rfe", *redo, "--n-jobs", "2"] for d in GEO]
                 + [["scripts/run_survey_e4_e5_stability.py", "--datasets", "gse37745", "--methods", "svm_rfe", "--wrappers", *redo, "--n-jobs", "2"],
                    ["scripts/run_survey_e4_e5_stability.py", "--datasets", TCGA, "--methods", "svm_rfe", "--wrappers", "--skip-e5a", "--skip-e5b",
                     *redo, "--n-jobs", "2"]],
                 [f"E1_classification/{d}" for d in C.DATASETS] + [f"E1b/{d}" for d in GEO] + ["E4_E5/gse37745", f"E4_E5/{TCGA}"],
                 requires=[f"E1_feature_selection/{d}/selected_genes.csv" for d in C.DATASETS]
                 + [f"E1_classification/{d}/classification_per_fold.csv" for d in C.DATASETS]
                 + [f"E1b/{d}/e1b_per_run.csv" for d in GEO] + ["E4_E5/gse37745/e4_selected_genes.csv", f"E4_E5/{TCGA}/e4_selected_genes.csv"],
                 requires_cfg=[(f"E1_feature_selection/{d}/config.json", "redo_methods", "svm_rfe") for d in C.DATASETS],
                 restore_from=["e1_phaseB_wrappers", "e1_phaseB_geo", "e1_phaseB_tcga", "fewshot_e1b", "e4e5_gse37745", "e4_tcga", "redo_svm_rfe_A"],
                 parallel=3, hours="2–3"))
    # ----------------------------------------------------------------- v3.1.7: panel công bố (QĐ32) — 05/11
    J.append(Job("panels", "P · 05/11", "Panel công bố Girard 2016 + Wilkerson 2013 (survey_v2/panels/published_panels.csv): E1 pha A "
                 "(panel cố định, gen thiếu ghi vào config) và pha B 5 bộ, E1b 4 cohort; nối vào kết quả đã có (phương pháp khác giữ nguyên)",
                 [_fs("--datasets", d, "--methods", *PUBLISHED_PANELS, "--redo-methods", *PUBLISHED_PANELS) for d in C.DATASETS],
                 [f"E1_feature_selection/{d}" for d in C.DATASETS] + [f"E1_classification/{d}" for d in C.DATASETS] + [f"E1b/{d}" for d in GEO],
                 finalize=[_clf("--datasets", d, "--no-references", "--methods", *PUBLISHED_PANELS, "--redo-methods", *PUBLISHED_PANELS) for d in C.DATASETS]
                 + [["scripts/run_survey_e1b_learning_curve.py", "--datasets", d, "--methods", *PUBLISHED_PANELS, "--redo-methods", *PUBLISHED_PANELS,
                     "--n-jobs", "2"] for d in GEO],
                 requires=[f"E1_feature_selection/{d}/selected_genes.csv" for d in C.DATASETS]
                 + [f"E1_classification/{d}/classification_per_fold.csv" for d in C.DATASETS] + [f"E1b/{d}/e1b_per_run.csv" for d in GEO],
                 requires_cfg=[(f"E1_feature_selection/{d}/config.json", "redo_methods", "svm_rfe") for d in C.DATASETS]
                 + [(f"E1_classification/{d}/config.json", "redo_methods", "svm_rfe") for d in C.DATASETS],
                 restore_from=["redo_svm_rfe_B"], parallel=4, hours="<0,5"))
    # ----------------------------------------------------------------- Ablation loại 4 gen IHC (09–10/11, khoá luận)
    J.append(Job("ablation_phaseA_geo", "4 · 10/11", "Ablation không IHC: pha A 4 cohort GEO", [_fs("--datasets", d, *abl) for d in GEO],
                 [f"E1_feature_selection_noIHC/{d}" for d in GEO], hours="3–6"))
    J.append(Job("ablation_phaseA_tcga", "4 · 10/11", "Ablation không IHC: pha A TCGA (5 lần lặp, 4 tiến trình)",
                 [_fs("--datasets", TCGA, "--repeats", str(r), *abl) for r in range(len(C.REPEAT_SEEDS))],
                 [f"E1_feature_selection_noIHC/{TCGA}"], finalize=[_fs("--datasets", TCGA, *abl)], hours="8–11"))
    J.append(Job("ablation_phaseB", "4 · 10/11", "Ablation không IHC: pha B 5 bộ",
                 [_clf("--datasets", d, "--selection-roots", "{RESULTS}/E1_feature_selection_noIHC", "--exclude-genes", *IHC_GENES,
                       "--out", "E1_classification_noIHC") for d in C.DATASETS],
                 [f"E1_classification_noIHC/{d}" for d in C.DATASETS],
                 requires=[f"E1_feature_selection_noIHC/{d}/selected_genes.csv" for d in C.DATASETS],
                 requires_cfg=[(f"E1_feature_selection_noIHC/{d}/config.json", "redo_methods", "svm_rfe") for d in C.DATASETS],
                 restore_from=["ablation_phaseA_geo", "ablation_phaseA_tcga", "redo_svm_rfe_A"], hours="3–6"))
    # ----------------------------------------------------------------- Bài 2 (Review 03/11, QĐ37): thử nghiệm sớm MVP
    J.append(Job("rq2_early", "M · 08/11", "Thử nghiệm sớm PC-04: TCGA chia 70 % nguồn giả / 30 % đích giả theo bệnh nhân; MVP (γ theo M5), "
                 "target-only, select-on-source; không kết luận H2",
                 [["scripts/run_rq2_mvp.py", "--early-tcga", "--save-scores", "--n-jobs", "4", "--out", "RQ2_mvp_early"]],
                 ["RQ2_mvp_early/tcga_early"], parallel=1, hours="<1"))
    J.append(Job("rq2_mvp", "M · 13/11", "RQ2 trên 4 cohort GEO (QĐ49–QĐ51): MVP (m5_pos, m5_cv, γ cố định, γ = 0 hai không gian gen), select-on-source, pooled, IHC-2, IHC-4; "
                 "có điểm từng mẫu; cổng bằng nhau với E1b, E7 ở bước verify",
                 [["scripts/run_rq2_mvp.py", "--datasets", *GEO, "--k", "2", "3", "4", "5", "10", "--gamma", "0", "0.5", "1", "2", "--save-scores", "--n-jobs", "4", "--out", "RQ2_mvp"]],
                 [f"RQ2_mvp/{d}" for d in GEO],
                 requires=[f"E1b/{d}/e1b_per_run.csv" for d in GEO] + [f"E7/{d}/e7_per_run.csv" for d in GEO],
                 requires_cfg=[(f"E1b/{d}/config.json", "methods_in_results", m) for d in GEO for m in ("l1_lr", "ihc_2", "ihc_4")],
                 restore_from=["e7", "panels"], gates=[["scripts/check_rq2_equality.py"]],
                 pin_versions=[f"E1b/{d}/config.json" for d in GEO] + [f"E7/{d}/config.json" for d in GEO], parallel=1, hours="2–3"))
    # ----------------------------------------------------------------- Độ nhạy RQ1 phần (b) (QĐ53b, Review 08/11 Mục 8.5)
    bas = RQ1_BAS_DATASET
    J.append(Job("rq1_sens_b", "S · 13/11", "Độ nhạy RQ1 (b): BAS → LUSC (gse30219_bas: L1-LR 8 mức k, IHC-4; B0, random-k) và lọc F-test 2.000 gen "
                 "trong fold (L1-LR, 3 cohort GPL570; B0 dùng lại E1_classification chính). Cần experiment-data-v3 có thư mục gse30219_bas",
                 [_fs("--datasets", bas, "--methods", "l1_lr", "ihc_4", "--out", "E1_feature_selection_bas")]
                 + [_fs("--datasets", d, "--methods", "l1_lr", "--prefilter", str(RQ1_FTEST_P0), "--prefilter-rule", "ftest",
                        "--out", "E1_feature_selection_ftest2000") for d in GPL570],
                 [f"E1_feature_selection_bas/{bas}", f"E1_classification_bas/{bas}"]
                 + [f"E1_feature_selection_ftest2000/{d}" for d in GPL570] + [f"E1_classification_ftest2000/{d}" for d in GPL570],
                 finalize=[_clf("--datasets", bas, "--selection-roots", "{RESULTS}/E1_feature_selection_bas", "--methods", "l1_lr", "ihc_4",
                                "--out", "E1_classification_bas")]
                 + [_clf("--datasets", d, "--selection-roots", "{RESULTS}/E1_feature_selection_ftest2000", "--methods", "l1_lr",
                         "--no-references", "--out", "E1_classification_ftest2000") for d in GPL570],
                 parallel=4, hours="~1"))
    # ----------------------------------------------------------------- RQ4 H4b + ablation RQ2 loại 4 gen IHC (Review 07/11 Bảng 8; 16–22/11)
    mvp_grid = ["--datasets", *GEO, "--k", "2", "3", "4", "5", "10", "--gamma", "0", "0.5", "1", "2", "--arms", "mvp", "--save-scores", "--n-jobs", "2"]
    J.append(Job("rq4_h4b_noihc", "R4 · 22/11", "H4b: MVP với π từ TCGA xáo nhãn (một hoán vị mỗi cohort) trên đúng các lần chạy của RQ2; "
                 "ablation RQ2: MVP và target-only khi loại 4 gen IHC khỏi nguồn và đích. Chỉ nhánh mvp; không cần khôi phục output",
                 [["scripts/run_rq2_mvp.py", *mvp_grid, "--shuffle-source-labels", "--out", "RQ2_mvp_shuffled"],
                  ["scripts/run_rq2_mvp.py", *mvp_grid, "--exclude-genes", *IHC_GENES, "--out", "RQ2_mvp_noIHC"]],
                 [f"RQ2_mvp_shuffled/{d}" for d in GEO] + [f"RQ2_mvp_noIHC/{d}" for d in GEO], parallel=2, hours="4–6"))
    return {j.name: j for j in J}


JOBS = build_jobs()


# ------------------------------------------------------------------------------------------------- helpers
def data_root() -> Path:
    root = os.environ.get("KLTN_DATA_ROOT")
    if not root:
        raise SystemExit("đặt KLTN_DATA_ROOT")
    return Path(root)


def results_root() -> Path:
    return data_root().joinpath(*C.RESULT_DIR)


def head_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def render(argv: list[str]) -> list[str]:
    res = str(results_root())
    return [a.replace("{RESULTS}", res) for a in argv]


def command(argv: list[str]) -> list[str]:
    return [sys.executable, *render(argv)]


def check_requires(job: Job) -> list[str]:
    """Tệp đầu vào còn thiếu + điều kiện config.json không đạt (v3.1.7, requires_cfg) — mô tả từng mục."""
    root = results_root()
    missing = [r for r in job.requires if not (root / r).exists()]
    for rel, key, expected in job.requires_cfg:
        path = root / rel
        if not path.exists():
            missing.append(f"{rel}: thiếu (cần {key} = {expected!r})")
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get(key)
        except (OSError, ValueError) as exc:
            missing.append(f"{rel}: không đọc được ({exc})")
            continue
        ok = (expected in value) if isinstance(value, (list, tuple, set)) else (value == expected)
        if not ok:
            missing.append(f"{rel}: {key} = {value!r}, cần {expected!r} (khôi phục đúng output và đúng thứ tự: {', '.join(job.restore_from)})")
    return missing


PINNED_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn")


def current_versions() -> dict:
    """Phiên bản PINNED_PACKAGES của môi trường đang chạy (None nếu chưa cài)."""
    from importlib import metadata
    out = {}
    for p in PINNED_PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def check_versions(job: Job, current: dict | None = None) -> tuple[list[str], dict]:
    """QĐ70 (Review 08/11): PINNED_PACKAGES của phiên phải trùng trường versions trong config.json của các đầu vào tham
    chiếu (E1b, E7 đã khôi phục). Trả (vấn đề, {gói: phiên bản cần cài}); bản cần cài chỉ có khi tham chiếu đọc được và
    thống nhất, để không đoán phiên bản."""
    if not job.pin_versions:
        return [], {}
    root = results_root()
    problems, expected = [], {}
    for rel in job.pin_versions:
        path = root / rel
        try:
            versions = json.loads(path.read_text(encoding="utf-8")).get("versions") or {}
        except (OSError, ValueError) as exc:
            problems.append(f"{rel}: không đọc được trường versions ({exc})")
            continue
        for p in PINNED_PACKAGES:
            v = versions.get(p)
            if not v:
                problems.append(f"{rel}: versions thiếu {p}")
            elif expected.setdefault(p, (v, rel))[0] != v:
                problems.append(f"{p}: tham chiếu không thống nhất ({expected[p][1]}: {expected[p][0]}; {rel}: {v})")
    if problems:
        return problems, {}
    current = current_versions() if current is None else current
    pins = {p: v for p, (v, _) in expected.items() if current.get(p) != v}
    problems = [f"{p}: phiên {current.get(p)}, tham chiếu {v} ({expected[p][1]})" for p, v in pins.items()]
    return problems, pins


TIMEOUT_CODE = 124


def run_parallel(cmds: list[list[str]], max_parallel: int, log_dir: Path, tag: str, deadline: float | None = None) -> list[int]:
    """Chạy các lệnh, tối đa max_parallel cùng lúc. Quá deadline (time.time()): dừng các tiến trình đang chạy (checkpoint
    đã ghi được giữ) và trả mã TIMEOUT_CODE cho các lệnh chưa xong — để notebook kịp lưu output trước giới hạn 12 giờ."""
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(enumerate(cmds))
    running: dict[int, tuple[subprocess.Popen, object, float]] = {}
    codes: dict[int, int] = {}
    env = os.environ.copy()
    while pending or running:
        if deadline is not None and time.time() > deadline:
            for i, (proc, log, _) in running.items():
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                log.close()
                codes[i] = TIMEOUT_CODE
                print(f"[stop  {tag}#{i}] hết thời gian của phiên — checkpoint đã ghi được giữ", flush=True)
            for i, _ in pending:
                codes[i] = TIMEOUT_CODE
            break
        while pending and len(running) < max(1, max_parallel):
            i, cmd = pending.pop(0)
            log = open(log_dir / f"{tag}_{i}.log", "w", encoding="utf-8")
            print(f"[start {tag}#{i}] {' '.join(cmd[1:])}  (log: {log.name})", flush=True)
            running[i] = (subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), log, time.time())
        time.sleep(5)
        for i in list(running):
            proc, log, t0 = running[i]
            if proc.poll() is not None:
                log.close()
                codes[i] = proc.returncode
                print(f"[done  {tag}#{i}] exit {proc.returncode} after {(time.time() - t0) / 60:.1f} min", flush=True)
                if proc.returncode != 0:
                    tail = (log_dir / f"{tag}_{i}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
                    print("    " + "\n    ".join(tail), flush=True)
                del running[i]
    return [codes[i] for i in sorted(codes)]


def write_pip_freeze(job: Job, log_dir: Path) -> Path | None:
    """v3.1.6: ghi `pip freeze` của môi trường chạy (Kaggle khác requirements.txt) để báo cáo đúng phiên bản."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        out = log_dir / f"pip_freeze_{job.name}.txt"
        txt = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120).stdout
        out.write_text(f"# python {sys.version.split()[0]}\n" + txt, encoding="utf-8")
        return out
    except Exception as exc:  # noqa: BLE001 — không được làm hỏng phiên vì bước ghi phụ
        print(f"[pip freeze] bỏ qua: {exc}", flush=True)
        return None


def verify(job: Job, commit: str | None) -> list[str]:
    problems = []
    root = results_root()
    for rel in job.outputs:
        cfg_path = root / rel / "config.json"
        if not cfg_path.exists():
            problems.append(f"{rel}: thiếu config.json")
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("traceable") is not True:
            problems.append(f"{rel}: traceable = {cfg.get('traceable')} {cfg.get('traceable_errors', '')}")
        if commit and cfg.get("commit_hash") != commit:
            problems.append(f"{rel}: commit_hash {cfg.get('commit_hash')} khác HEAD {commit}")
        if "calibration_ok" in cfg and cfg.get("calibration_ok") is False:
            problems.append(f"{rel}: calibration_ok = false (QĐ18: không được trích)")
    for gate in job.gates:
        code = subprocess.run(command(gate), cwd=REPO).returncode
        if code != 0:
            problems.append(f"cổng {gate[0]}: mã {code}")
    return problems


# ------------------------------------------------------------------------------------------------- E2/E3 merge
E2E3_TABLES = ("e2_cost_point.csv", "e3_scaling_n.csv", "e3_scaling_p.csv", "e3_wrapper_evals.csv", "e3_exponents.csv")


def _read_csv_or_none(path: Path):
    """CSV -> DataFrame; None nếu thiếu hoặc rỗng (kể cả tệp chỉ có BOM + xuống dòng mà pandas báo EmptyDataError)."""
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return frame if len(frame) else None


def merge_e2e3(parts=tuple(f"E2_E3_{p}" for p in E3_PARTS), ref: str = "E2_E3_ref", out_name: str = "E2_E3") -> Path:
    """E2 chính thức = phiên tham chiếu (mọi phương pháp trong một phiên); E3 = gộp các phần đã hiệu chuẩn với phiên đó.
    Mọi phần phải traceable, calibration_ok = true và cùng commit với phiên tham chiếu. config.json của thư mục gộp được
    stamp bằng provenance của chính lần gộp (M18: không gán traceable bằng tay)."""
    from survey_v2.runlog import start_run, write_config
    root = results_root()
    out = root / out_name
    out.mkdir(parents=True, exist_ok=True)
    sources = (ref, *parts)
    cfgs, problems = {}, []
    for p in sources:
        cfg_path = root / p / "config.json"
        if not cfg_path.exists():
            raise SystemExit(f"thiếu {cfg_path}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfgs[p] = cfg
        if cfg.get("traceable") is not True:
            problems.append(f"{p}: traceable = {cfg.get('traceable')}")
        if p != ref and cfg.get("calibration_ok") is not True:
            problems.append(f"{p}: calibration_ok = {cfg.get('calibration_ok')}")
    commits = {c.get("commit_hash") for c in cfgs.values()}
    if len(commits) != 1:
        problems.append(f"các phần khác commit: {sorted(map(str, commits))}")
    for p, cfg in cfgs.items():                                  # v3.1.6 (QĐ27): phiên đo ở bản cũ không được gộp
        if cfg.get("timing_tracemalloc") is not False:
            problems.append(f"{p}: đo thời gian ở bản trước v3.1.6 (tracemalloc bật) — chạy lại phiên này")
        if p == ref and cfg.get("isolate_memory") and cfg.get("peak_source") != "VmHWM":
            problems.append(f"{p}: bộ nhớ tiến trình con đo bằng ru_maxrss kế thừa (trước v3.1.6) — chạy lại phiên tham chiếu")
    if problems:
        raise SystemExit("KHÔNG gộp:\n  " + "\n  ".join(problems))
    prov = start_run(__file__, params={"merge": "e2e3", "ref": ref, "parts": list(parts)})
    # E2: từ phiên tham chiếu; E2 của các phần (nếu có, vd. chạy không --skip-e2) giữ riêng để đối chiếu độ lệch
    ref_e2 = root / ref / "e2_cost_point.csv"
    if not ref_e2.exists():
        raise SystemExit(f"thiếu {ref_e2} (phiên tham chiếu phải đo E2)")
    e2 = pd.read_csv(ref_e2)
    e2.insert(0, "session", ref)
    for col in ("peak_mb", "peak_mb_all", "maxrss_mb"):          # v3.1.6: cột tracemalloc / ru_maxrss của bản cũ không dùng
        if col in e2.columns:
            e2 = e2.drop(columns=col)
    e2.to_csv(out / "e2_cost_point.csv", index=False, encoding="utf-8-sig")
    part_e2 = []
    for p in parts:
        frame = _read_csv_or_none(root / p / "e2_cost_point.csv")
        if frame is not None:
            frame.insert(0, "session", p)
            part_e2.append(frame)
    if part_e2:
        pd.concat(part_e2, ignore_index=True).to_csv(out / "e2_cost_point_parts.csv", index=False, encoding="utf-8-sig")
    # E3: gộp các phần
    for name in E2E3_TABLES[1:]:
        frames = []
        for p in parts:
            frame = _read_csv_or_none(root / p / name)           # v3.1.6: tệp rỗng (chỉ BOM) -> bỏ qua
            if frame is not None:
                frame.insert(0, "session", p)
                frames.append(frame)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out / name, index=False, encoding="utf-8-sig")
    for p in sources:
        for name in ("calibration.csv", "calibration_check.json", "machine_config.json"):
            if (root / p / name).exists():
                (out / f"{p}_{name}").write_bytes((root / p / name).read_bytes())
    if (root / ref / "machine_config.json").exists():
        (out / "machine_config.json").write_bytes((root / ref / "machine_config.json").read_bytes())
    first = cfgs[ref]
    merged = {"experiment": "E2_E3", "e2_from": ref, "e3_merged_from": list(parts), "merged_by": "scripts/kaggle_jobs.py merge-e2e3",
              "sources_commit_hash": first.get("commit_hash"),
              "calibration_ok": all(cfgs[p].get("calibration_ok") is True for p in parts),
              "calibration": {p: c.get("calibration") for p, c in cfgs.items()},
              "machine": {p: c.get("machine") for p, c in cfgs.items()},
              "methods": {p: c.get("methods") for p, c in cfgs.items()}, "dataset": first.get("dataset"), "split": first.get("split"),
              "k": first.get("k"), "k_e3": first.get("k_e3"), "p_fixed": first.get("p_fixed"), "n_grid": cfgs[parts[0]].get("n_grid"),
              "p_grid": cfgs[parts[0]].get("p_grid"), "e3_max_p": cfgs[parts[0]].get("e3_max_p"), "repeats": first.get("repeats"),
              "isolate_memory": first.get("isolate_memory"), "peak_source": first.get("peak_source"),
              "timing_tracemalloc": first.get("timing_tracemalloc", True),
              "note": "E2 = phiên tham chiếu (một phiên, mọi phương pháp); E3 = các phần đo ở phiên riêng, hiệu chuẩn với phiên tham chiếu (QĐ18); cột session ghi phần."}

    write_config(out, prov, merged)
    print("gộp xong:", out)
    return out


# ------------------------------------------------------------------------------------------------- CLI
def cmd_plan():
    rows = [(j.batch, j.name, j.hours, ", ".join(j.restore_from) or "—", j.what) for j in JOBS.values()]
    width = max(len(r[1]) for r in rows)
    for batch, name, hours, restore, what in rows:
        print(f"{batch:12s} {name:{width}s}  {hours:>10s} h  khôi phục: {restore}\n{'':12s} {'':{width}s}  {what}")


def cmd_show(job: Job):
    print(f"{job.name} — {job.what}\nđợt {job.batch}; song song {job.parallel}; exclusive {job.exclusive}; ước lượng {job.hours} giờ")
    print("khôi phục (gắn output làm Kaggle Input, ĐÚNG THỨ TỰ — zip sau ghi đè zip trước):", ", ".join(job.restore_from) or "—")
    print("cần có:", ", ".join(job.requires) or "—")
    if job.requires_cfg:
        print("config.json cần đạt:", "; ".join(f"{r}: {k} ∋ {v!r}" for r, k, v in job.requires_cfg))
    if job.pin_versions:
        print(f"phiên bản {', '.join(PINNED_PACKAGES)} phải trùng versions của:", ", ".join(job.pin_versions))
    for s in job.shards:
        print("  python", " ".join(render(s)) if os.environ.get("KLTN_DATA_ROOT") else " ".join(s))
    for s in job.finalize:
        print("  [finalize] python", " ".join(render(s)) if os.environ.get("KLTN_DATA_ROOT") else " ".join(s))
    print("đầu ra kiểm tra:", ", ".join(job.outputs) or "—")
    for g in job.gates:
        print("  [cổng ở bước verify] python", " ".join(g))


def cmd_run(job: Job, parallel: int | None, log_dir: Path, max_hours: float | None = None) -> int:
    missing = check_requires(job)
    if missing:
        print("THIẾU đầu vào hoặc đầu vào sai bản (gắn output các việc " + ", ".join(job.restore_from)
              + " làm Input, đúng thứ tự):\n  " + "\n  ".join(missing))
        return 2
    if job.name == "panels" and not PUBLISHED_PANELS:
        print("panels: survey_v2/panels/published_panels.csv chưa có panel nào")
        return 2
    bad_versions, pins = check_versions(job)
    if bad_versions:
        print("PHIÊN BẢN THƯ VIỆN KHÁC THAM CHIẾU (QĐ70) — không chạy:\n  " + "\n  ".join(bad_versions))
        if pins:
            print("cài đúng phiên bản rồi chạy lại JOB — notebook rq2_mvp: dán vào ô cấu hình\n  PIP_PINS = " + repr(pins)
                  + "\n  (tương đương pip install " + " ".join(f"{p}=={v}" for p, v in pins.items()) + ")")
        return 2
    commit = head_commit()
    deadline = time.time() + max_hours * 3600 if max_hours else None
    print(f"JOB {job.name} | commit {commit} | data {data_root()} | giới hạn {max_hours or '—'} giờ", flush=True)
    write_pip_freeze(job, log_dir)
    n_par = 1 if job.exclusive else (parallel or job.parallel)
    sequential = job.name == "e1_phaseB_tcga"      # gộp pha A rồi mới pha B
    codes = run_parallel([command(s) for s in job.shards], 1 if sequential else n_par, log_dir, job.name, deadline)
    if TIMEOUT_CODE in codes:
        print(f"TIMEOUT {job.name}: chưa xong trong phiên này. Lưu output, gắn nó làm Input rồi chạy lại cùng JOB (tự tiếp tục).")
        return TIMEOUT_CODE
    if any(codes):
        print(f"LỖI: shard thoát khác 0: {codes}. Output đã có vẫn được giữ; chạy lại cùng JOB sẽ tiếp tục.")
        return 1
    if job.finalize:
        codes = run_parallel([command(s) for s in job.finalize], 1, log_dir, job.name + "_finalize", deadline)
        if TIMEOUT_CODE in codes:
            print(f"TIMEOUT {job.name} ở bước gộp: chạy lại cùng JOB.")
            return TIMEOUT_CODE
        if any(codes):
            return 1
    problems = verify(job, commit)
    if problems:
        print("VERIFY KHÔNG ĐẠT:\n  " + "\n  ".join(problems))
        return 3
    print(f"JOB_OK {job.name}: " + ", ".join(job.outputs))
    return 0


def cmd_status():
    root = results_root()
    for j in JOBS.values():
        have = [o for o in j.outputs if (root / o / "config.json").exists()]
        mark = "xong" if j.outputs and len(have) == len(j.outputs) else (f"{len(have)}/{len(j.outputs)}" if j.outputs else "—")
        print(f"{j.batch:12s} {j.name:24s} {mark}")
    # v3.1.7: bản nào đang nằm trong results/ (commit, redo_methods) — để thấy ngay khi khôi phục sai thứ tự
    if root.exists():
        print("thư mục kết quả hiện có (commit, redo_methods / methods_in_results):")
        for cfg_path in sorted(root.glob("*/config.json")) + sorted(root.glob("*/*/config.json")):
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            rel = cfg_path.parent.relative_to(root).as_posix()
            extra = ""
            if cfg.get("redo_methods"):
                extra += f" redo={cfg['redo_methods']}"
            if cfg.get("methods_in_results"):
                extra += f" n_methods={len(cfg['methods_in_results'])}"
            if "timing_tracemalloc" in cfg:
                extra += f" tracemalloc={cfg['timing_tracemalloc']} peak={cfg.get('peak_source')}"
            print(f"  {rel:50s} {str(cfg.get('commit_hash'))[:7]}{extra}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "list", "show", "run", "verify", "status", "merge-e2e3"])
    ap.add_argument("job", nargs="?")
    ap.add_argument("--parallel", type=int, default=None, help="số tiến trình song song tối đa (mặc định theo việc)")
    ap.add_argument("--log-dir", type=Path, default=Path(os.environ.get("KLTN_LOG_DIR", "logs")))
    ap.add_argument("--max-hours", type=float, default=None, help="dừng có kiểm soát sau số giờ này (Kaggle: 11)")
    args = ap.parse_args(argv)
    if args.command in ("plan", "list"):
        cmd_plan()
        return 0
    if args.command == "status":
        cmd_status()
        return 0
    if args.command == "merge-e2e3":
        merge_e2e3()
        return 0
    if args.job not in JOBS:
        ap.error(f"JOB phải là một trong: {', '.join(JOBS)}")
    job = JOBS[args.job]
    if args.command == "show":
        cmd_show(job)
        return 0
    if args.command == "verify":
        problems = verify(job, head_commit())
        print("VERIFY_OK" if not problems else "VERIFY KHÔNG ĐẠT:\n  " + "\n  ".join(problems))
        return 0 if not problems else 3
    return cmd_run(job, args.parallel, args.log_dir, args.max_hours)


if __name__ == "__main__":
    sys.exit(main())
