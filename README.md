# RNA-Seq FS Survey – chọn gen có ràng buộc ngân sách cho dữ liệu biểu hiện gen ung thư phổi

Mã nguồn khóa luận: chọn/trích xuất đặc trưng, phân loại, permutation test (bài toán phụ: giai đoạn sớm/muộn) và RQ0′–RQ4 (bài toán chính: LUAD vs LUSC, chuyển giao TCGA → cohort GEO).

**Kho này là nơi duy nhất chứa code.** Dữ liệu và kết quả nằm ở thư mục dữ liệu riêng (vd. `D:\KLTN`), không đưa lên Git. Mọi kết quả phải sinh từ một commit đã push, để commit hash ghi trong `config.json` kiểm chứng được trên GitHub (Review 20/10/2026, O3, §7.7).

## Cấu trúc

```
RNA-Seq-FS-Survey/
├── scripts/                     code chạy trên máy (Python)
│   ├── provenance.py            ghi nguồn gốc mỗi lần chạy; dừng nếu không truy vết được
│   ├── run_rq0_prime.py         RQ0′: trần hiệu năng trên 4 cohort GEO (CV 5×5)
│   ├── build_rq0_prime_workbook.py
│   ├── rebuild_gse30219_r5.py   dựng lại GSE30219 sau khi sửa nhãn (R5): 85 LUAD / 61 LUSC
│   ├── build_perm_test_data.py  dữ liệu cho permutation test v3
│   ├── patch_v2_tcga_stage.py   vá nhãn giai đoạn TCGA v2 từ GDC
│   ├── aggregate_phase0_v3.py, aggregate_phase1_results.py, stage_split_metrics.py, organize_fs_phase1.py
│   ├── build_review_completion_artifacts.py, build_review_documents.py
│   ├── run_survey_e1.py, run_survey_e1_wrappers.py, run_survey_e2_e3_timing.py, run_survey_e4_e5_stability.py
│   ├── make_survey_tables.py, run_p0_perm_v3.py, aggregate_p0_decision.py, run_sanity_auroc.py
│   ├── smoke_test_survey_v2.py  chạy thử toàn bộ chuỗi survey trên dữ liệu giả
│   ├── PHASE1_REPRODUCTION.md
│   └── _archive/                script cũ, không dùng
├── survey_v2/                   gói Python của bộ thực nghiệm survey v2 (E1–E6, P0, sanity) — xem docs/survey_v2/
├── tests/                       python -m pytest tests
├── notebooks/
│   ├── phase_0/                 dữ liệu v1/v2/GEO, FS/FE và phân loại Phase 0, permutation v3, B0 v3
│   ├── phase_1/                 benchmark Phase 1 (LUAD vs LUSC) – ĐÓNG BĂNG (O13)
│   └── _archive/                notebook cũ, không dùng
├── docs/notebook_manifest.md    trạng thái từng notebook: đang dùng / đóng băng / cũ
└── docs/survey_v2/HUONG-DAN-DOI-CODE.md   hướng dẫn đội code cho survey v2 (thiết kế, lệnh, tiêu chí chấp nhận)
```

## Cài đặt

```powershell
git clone https://github.com/datachain-uit/RNA-Seq-FS-Survey.git
cd RNA-Seq-FS-Survey
pip install -r requirements.txt
$env:KLTN_DATA_ROOT = "D:\KLTN"     # thư mục chứa experiment-data*/, results/ ...
```

`KLTN_DATA_ROOT` cho script biết dữ liệu ở đâu. Thiếu biến này, script tìm dữ liệu ngay trong thư mục kho.

## Quy trình mỗi lần chạy

1. Sửa code trong kho → `git commit` → `git push`.
2. Chạy script từ thư mục kho, ví dụ `python scripts/run_rq0_prime.py`.
3. Script gọi `provenance()` trước khi tính toán:
   - **dừng** nếu thư mục không phải git clone hoặc còn file đã theo dõi bị sửa mà chưa commit;
   - ghi vào `config.json`/`provenance.json` cạnh kết quả: commit hash, script và tham số, phiên bản thư viện, máy, SHA-256 của dữ liệu đầu vào.

Chỉ trong trường hợp bất khả kháng mới chạy với `$env:KLTN_ALLOW_UNTRACKED = "1"`; kết quả khi đó ghi `commit_hash: null` và không được trích vào survey hay luận văn.

Notebook chạy trên Kaggle/Colab: clone kho ở đúng commit rồi dùng `scripts/provenance.py` (ô mẫu ở `docs/notebook_manifest.md`, mục 4).

## Dữ liệu (ở `KLTN_DATA_ROOT`, không có trong kho)

| Thư mục | Nội dung |
|---|---|
| `experiment-data-v2/` | TCGA LUAD/LUSC (1.018 bệnh nhân), GSE68465 chỉ mẫu u, sanity Normal vs u; có `DATA_CARD.md` |
| `experiment-data-geo-target/` | cohort đích GSE10245, GSE30219 (`gse30219_r5_fixed`), GSE37745, GSE42127 |
| `experiment-data-perm/` | dữ liệu permutation test v3 |
| `results/` | kết quả theo phase |

## Survey v3.1 (E0–E7, dùng chung với RQ1 khoá luận) — 29/10/2026 (QĐ15, QĐ18–QĐ20)

v3.1 = gói v3 (26/10) ghép lên HEAD `2508a2a` của đội code: giữ `dataset_path()`, `dataset_seed_id()`, `build_v3_splits.py`
(phần few-shot sửa thành 4 tập con lồng nhau), layout Kaggle `experiment-data-geo-target/gse30219`; thêm `KLTN_STRICT_N`.
Chi tiết và việc phải làm trước: mục "v3.1" đầu `docs/survey_v2/HUONG-DAN-DOI-CODE.md`.
v3.1.6 (31/10/2026, QĐ25–QĐ28, sau đợt 1): SVM-RFE có `random_state` (chọn lại bằng `--redo-methods svm_rfe`, việc `redo_svm_rfe_A/B`);
abess bỏ cột hằng; đo thời gian E2/E3 không tracemalloc và bộ nhớ tiến trình con bằng VmHWM (E2/E3 chạy lại); `requires` cho E4/E5;
`merge-e2e3` bỏ qua CSV rỗng. Mục "v3.1.6" đầu HUONG-DAN-DOI-CODE.md.
v3.1.7 (02/11/2026, QĐ30–QĐ32, sau bổ sung đợt 1): seed bootstrap của `make_survey_tables.py` tái lập (CRC32 của khoá, không `hash()`),
cờ chuyển giao E7 theo Wilcoxon + Holm; bảng ablation IHC (`e1_ablation_noihc*`); việc `panels` cho Girard 2016 / Wilkerson 2013;
`restore_from` = thứ tự giải nén (mới nhất cuối) và `requires_cfg` chặn khôi phục sai bản; nối CSV theo tiêu đề (`append_csv_aligned`).
Mục "v3.1.7" đầu HUONG-DAN-DOI-CODE.md.

Bài survey theo cùng bài toán với khoá luận (LUAD vs LUSC; nguồn TCGA + 4 cohort GEO). Bộ thực nghiệm nằm trong `survey_v2/`
và `scripts/`; hướng dẫn đầy đủ (mục đích từng thí nghiệm, lệnh, đầu ra, tiêu chí chấp nhận, bàn giao) ở
[`docs/survey_v2/HUONG-DAN-DOI-CODE.md`](docs/survey_v2/HUONG-DAN-DOI-CODE.md). Bài toán giai đoạn (P0, GSE68465) không còn thuộc survey.

```powershell
python -m pytest tests -q                              # kiểm tra đơn vị + 7 test đầu-cuối trên dữ liệu giả
python scripts/smoke_test_survey_v2.py --n-jobs 4      # chạy thử toàn bộ chuỗi E0–E7 (≈ 10–20 phút, dữ liệu giả)
python scripts/build_e1_repeated_splits.py             # bản chia 5×5 theo bệnh nhân cho 5 bộ
python scripts/build_fewshot_splits.py                 # bản chia few-shot (5 split × 4 tập con × n_T) cho 4 cohort GEO
python scripts/run_sanity_auroc.py                     # Y10: đối chứng dương
python scripts/run_survey_e0_signal_check.py           # E0: hoán vị + random-k cho 5 bộ
python scripts/run_e1_feature_selection.py             # E1 pha A: 11 phương pháp + panel IHC, 5 bộ, k ∈ {1..50}
python scripts/run_e1_wrappers_all_cohorts.py          # E1 pha A: GA, BPSO (3 cohort GPL570, 3×5, k ∈ {5,10,50})
python scripts/run_e1_classification.py --secondary    # E1 pha B: LR-L2 (+ RF, SVM ở k ∈ {5,25}), B0, random-k
python scripts/run_survey_e1b_learning_curve.py        # E1b: đường cong học few-shot (= nhánh target-only của RQ2)
python scripts/run_survey_e7_transfer_naive.py         # E7: source-only / select-on-source / pooled (= baseline D3 của RQ2)
python scripts/run_survey_e2_e3_timing.py --wrappers   # E2 + E3 (TCGA, một máy cố định)
python scripts/run_survey_e4_e5_stability.py           # E4 (TCGA, GSE37745) + E5 (GSE37745)
python scripts/make_survey_tables.py                   # bảng, hình, k* gộp (QĐ11), E6, manifest truy vết
```

Kết quả vào `$KLTN_DATA_ROOT/results/survey_v2/<E>/...`, mỗi thư mục có `config.json` (commit hash, SHA-256 dữ liệu và bản chia,
`traceable`, `gene_space`). Script dừng nếu chưa commit + push (Y13); `KLTN_ALLOW_UNPUSHED=1` chỉ để chạy thử.
