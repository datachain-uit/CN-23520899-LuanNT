# Hướng dẫn đội code — Thực nghiệm survey v3.1 (E0–E7, dùng chung với RQ1 khoá luận)

Phiên bản 02/11/2026 (v3.1.7; v3.1 = bản v3 của Review 26/10 ghép lên HEAD `2508a2a` của đội code, QĐ18–QĐ20 của Review 29/10;
v3.1.1–v3.1.5 = QĐ21–QĐ24; v3.1.6 = QĐ25–QĐ28 của Review 31/10; v3.1.7 = QĐ30–QĐ32 của Review 02/11).
Áp dụng cho gói `survey_v2/` + `scripts/` trên nhánh `survey-v2-code`. Người nhận: đội code (LuanNT). Người viết bài: GVHD.
Mọi thắc mắc về *thiết kế* hỏi GVHD trước khi đổi code.

---

## v3.1.7 — sau bổ sung đợt 1 (Review 02/11/2026, QĐ30–QĐ32): đọc trước, làm trong 03–05/11

Bổ sung đợt 1 (9 phiên ở `6ea47e8`, 01–02/11) được nghiệm thu: svm_rfe chọn lại tái lập (40/40 REUSE), E2/E3 đo lại không
tracemalloc + VmHWM, hiệu chuẩn đạt lần đầu ở cả ba phiên, 12 phương pháp còn lại trùng từng byte với đợt 1. Hai phát hiện của
đội code (HANDOVER 04/11, Phát hiện 2 và 4) là đúng và được sửa trong bản này; GVHD phát hiện thêm một lỗi nối CSV (Y31). Panel
công bố chưa chạy (CSV điền sau commit chạy phiên) nên bảng chính thức còn thiếu hàng `panel_*` và bảng ablation IHC.

| # | Vấn đề | Ảnh hưởng | Sửa trong v3.1.7 | Phải chạy lại |
|---|---|---|---|---|
| QĐ30 | `make_survey_tables.py` lấy seed bootstrap từ `hash()` của Python (ngẫu nhiên theo tiến trình, PYTHONHASHSEED) — Phát hiện 4 | cận CI của `e1b_learning_curves`, `e7_transfer_table` lệch ≤ 0,008 giữa hai lần chạy cùng đầu vào; cờ `positive_transfer` (= cận dưới CI > 0) đổi ở 1 ô (GSE30219, select-on-source, n_T = 10, k = 5) | seed = CRC32 của khoá (cohort, method, k, n_T); cờ `positive/negative_transfer` theo kiểm định đặt trước (Wilcoxon + Holm, α = 0,05) và dấu của `diff_mean`; CI chỉ mô tả (`ci_excludes_zero`); Wilcoxon gọi với tham số mặc định ghi tường minh | `tables` (0,04 giờ; sau `panels`) |
| QĐ31 | `tables/` không có bảng ablation IHC (E1_classification_noIHC không phải đầu vào của script) — HANDOVER 3.7 | bảng ablation của khoá luận (hạn 10/11) phải tính tay | `e1_ablation_noihc.csv` (mọi k) và `e1_ablation_noihc_table.csv/.tex` (k ∈ {2, 5, 10}): ΔAUROC = không IHC − đầy đủ ghép cặp theo fold, CI 95 % Nadeau–Bengio, số gen IHC trung bình trong panel đầy đủ | `tables` |
| QĐ32 | `published_panels.csv` điền ở `b0a1844` (sau `6ea47e8`) nên không có phiên nào chạy `panel_girard2016`, `panel_wilkerson2013` (G-14, R1-04) | bảng E1/E1b chính thức thiếu hai panel công bố | việc `panels` (notebook `kaggle_panels.ipynb`): pha A (panel cố định, gen thiếu ghi vào config), pha B `--no-references`, E1b, `--redo-methods` chỉ hai panel; `make_survey_tables` có nhãn/thứ tự cho `panel_*` (so với B0 được; không so với random-k vì k = 42/52 ngoài lưới) | `panels` (< 0,5 giờ; khôi phục `redo_svm_rfe_B`) rồi `tables` |
| O28 | `restore_from` của `redo_svm_rfe_B` liệt kê `redo_svm_rfe_A` đầu tiên trong khi notebook giải nén theo thứ tự và zip sau ghi đè zip trước — Phát hiện 2 (đội code đã tự đặt cuối, kết quả đúng) | nếu làm theo thứ tự in ra, svm_rfe mới bị ghi đè im lặng | `restore_from` = thứ tự giải nén, output mới nhất cuối; `requires_cfg`: `run` dừng (mã 2) nếu config.json của đầu vào không đúng bản (vd. E1_feature_selection chưa `redo_methods ∋ svm_rfe`, E2_E3 còn tracemalloc); `status` in commit + redo_methods của từng thư mục đã khôi phục | không |
| Y31 | pha B nối dòng vào `classification_per_fold.csv` bằng `to_csv(mode="a")` không khớp tiêu đề: dòng panel nối sau dòng random-k (không có cột `n_draws`) bị ghi theo vị trí → `n_features_vs_k` rơi vào cột `n_draws` (ga/bpso ở 3 cohort GPL570, svm_rfe ở 5 bộ). Các cột chỉ số đứng trước nên **không đổi** (GVHD đã kiểm) | hai cột sổ sách sai ở 1.950 dòng (svm_rfe 5 bộ, GA/BPSO 3 cohort); không ảnh hưởng bảng | `append_csv_aligned` (nối theo tiêu đề; dùng cho pha B, E1b, E4/E5, E7); `normalise_per_fold` sửa dòng cũ khi đọc lại, ghi `legacy_rows_realigned` vào config | không riêng (tự sửa khi `panels` chạy pha B) |

**Không phải chạy lại:** mọi phiên đã có (đợt 1 và bổ sung đợt 1). `tables_results.zip` ngày 02/11 vẫn đúng cho mọi cột trừ
cận CI bootstrap của E1b/E7 và cờ chuyển giao; bản `tables` mới thay thế nó.

**Áp gói.** Trên `survey-v2-code` tại `0e03837`: `git am --3way survey-v3.1.7-<hash>.patch`, `python -m pytest tests -q`
(≈ 218 test nhanh + 7 test đầu-cuối; `-m "not slow"` ≈ 30 s), `python scripts/make_kaggle_notebooks.py --check` (11 notebook),
push, đặt `REPO_COMMIT` mới. Chạy lại `make_survey_tables.py` hai lần trên cùng đầu vào với `PYTHONHASHSEED` khác nhau phải cho
`tables/*.csv` giống hệt (test đầu-cuối đã kiểm).

**Thứ tự phiên (03–05/11):** (1) `panels` — khôi phục `redo_svm_rfe_B`; (2) `tables` — khôi phục đúng thứ tự
`e0, sanity, e7, e2e3_merge, ablation_phaseB, redo_svm_rfe_B, panels` (`kaggle_jobs.py show tables`). Bàn giao như HANDOVER 04/11
kèm: diff `tables/*.csv` mới so với bản 02/11 (chỉ khác CI/cờ E7 và các hàng/bảng mới), config.json có `legacy_rows_realigned`.

---

## v3.1.6 — sau đợt 1 (Review 31/10/2026, QĐ25–QĐ28): đọc trước, làm trong 02–06/11

Đợt 1 (17 phiên, 29–30/10) được nghiệm thu về quy trình: mọi `config.json` traceable, đúng n sau QC, E0 đạt 5/5, sanity đạt.
Khi đối chiếu số liệu GVHD phát hiện ba lỗi đo/tái lập **trong code GVHD giao** (không phải lỗi vận hành của đội code) và một
lỗi vận hành nhỏ. Bản v3.1.6 sửa cả bốn; việc phải chạy lại được giới hạn tối đa.

| # | Lỗi | Ảnh hưởng | Sửa trong v3.1.6 | Phải chạy lại |
|---|---|---|---|---|
| QĐ25 | `rank_svm_rfe` dùng `LinearSVC(dual="auto")` không `random_state`; p > n → bộ giải dual xáo trộn ngẫu nhiên → hai phiên cho hai panel (30/120 panel TCGA khác nhau, HANDOVER 31/10) | mọi kết quả svm_rfe (E1 A/B, E1b, E4/E5, ablation A) không tái lập; E4 của svm_rfe lẫn nhiễu bộ giải | `random_state = seed` (cả `svm_linear` pha B và LR fitness của wrapper: primal, tất định, chỉ để an toàn) | **chỉ svm_rfe**: `redo_svm_rfe_A` (pha A 5 bộ + ablation A) rồi `redo_svm_rfe_B` (pha B, E1b, E4/E5) với `--redo-methods svm_rfe` (≈ 3–4 giờ Kaggle) |
| QĐ26 | abess 0.4.11 báo `OverflowError: The variable … is constant` khi ma trận có cột hằng (E3 p = toàn bộ gen TCGA: 418 gen hằng) | e3_p1 không chạy được ở p = 19.938 | `select_abess` bỏ cột sd = 0 rồi ánh xạ chỉ số về; `info["n_constant_dropped"]` | chỉ e3_p1 (đang thiếu) |
| QĐ27 | (a) `measure()` bật `tracemalloc` trong lúc đo → wall time tăng tuỳ phương pháp (đo lại trên máy GVHD: ReliefF ×5,7; MI ×3,0; RF ×2,5; L1 ×1,1; F-test ×1,0). (b) `ru_maxrss` của tiến trình con **kế thừa đỉnh RSS của tiến trình cha** trên Linux → `maxrss_child_mb` = `maxrss_mb` của cha trong mọi dòng E2_E3_ref, `maxrss_isolated_mb` tăng đơn điệu (0 → 43,8 → 551 MB) | bảng E2 (thời gian + bộ nhớ) và số mũ E3 của e2_ref, e3_p2, e3_p3 **không được trích** (M19) | không tracemalloc; khởi động trên lát nhỏ; tiến trình con đo `VmHWM` (`peak_rss_mb()`); `merge-e2e3` từ chối phiên đo ở bản cũ | `e2_ref`, `e3_p1`, `e3_p2`, `e3_p3`, `e2e3_merge` (≈ 10–11 giờ, chạy song song 3–4 phiên) |
| QĐ28 | `e4e5_gse37745`, `e4_tcga` không khai `requires` → E5c bị bỏ qua im lặng khi thiếu pha A (hai phiên "lan1-thieu-E5c"); `merge-e2e3` lỗi `EmptyDataError` với `e3_wrapper_evals.csv` rỗng (4 byte) | tốn 2 phiên; merge không chạy được | `requires` cho e4e5/e4_tcga/e7/tables; merge bỏ qua CSV rỗng; `run` ghi `logs/pip_freeze_<JOB>.txt`; `config.json` ghi `resumed_from` khi tiếp tục ở commit khác | không |

**Không phải chạy lại:** E0, sanity, E1 pha A/B của 12 phương pháp còn lại và wrapper, E1b, E7, E4/E5 (trừ dòng svm_rfe), ablation A.
Các zip đợt 1 là đầu vào khôi phục của các việc redo (xem `python scripts/kaggle_jobs.py show redo_svm_rfe_A`).

**Áp gói.** Trên `survey-v2-code` tại `68e7c7d`: `git am --3way survey-v3.1.6-<hash>.patch`, `python -m pytest tests -q`
(≈ 200 test + 7 test đầu-cuối, `-m "not slow"` ≈ 30 s), `python scripts/make_kaggle_notebooks.py --check`, push, rồi đặt `REPO_COMMIT`
mới trong các notebook Kaggle. Notebook mới: `notebooks/kaggle/kaggle_redo_svm_rfe.ipynb`.

**Thứ tự phiên (02–06/11):** (1) `redo_svm_rfe_A` → `redo_svm_rfe_B` (04/11); (2) `e2_ref` mới → `e3_p1`, `e3_p2`, `e3_p3` →
`e2e3_merge` (06/11); (3) `ablation_phaseB` (khôi phục output `redo_svm_rfe_A`); (4) `tables` (sau cùng). Bảng/hình trích vào bài
chỉ lấy từ `tables` chạy sau (1)–(3) và sau khi có `published_panels.csv` (quá hạn 31/10 — xem Review 31/10 mục 6).

---

## v3.1 — điều gì khác bản v3 (26/10) và phải làm gì trước (đọc trước mọi mục khác)

**Vì sao có v3.1.** Gói v3 (patch trên `570bace`) chưa được áp; từ 26 đến 27/10 đội code commit thêm 6 commit (`511cbf2` → `2508a2a`)
sửa `config.py`, `data.py`, hai script E1 và thêm `build_v3_splits.py` + 3 tệp test. v3.1 là gói v3 ghép lên `2508a2a` để không mất
việc đã làm, đã chạy `pytest` (89 test, kể cả 3 tệp test của đội code) và smoke test đầu-cuối trên dữ liệu giả.

**Những gì v3.1 giữ của đội code** (không phải sửa notebook/test đang có):
- `survey_v2.data.dataset_path(root, name)` và `config.dataset_seed_id(name)` (= 100 + thứ tự cohort; `cohort_id()` của v3 nay
  trả về cùng giá trị) — seed của pha A trên TCGA vì thế **không đổi** so với lần chạy 26–28/10, đủ điều kiện làm kiểm tra bằng nhau (QĐ19).
- Thư mục dữ liệu theo layout Kaggle Input `experiment-data-v3` của DATA_AUDIT 27/10: `experiment-data-geo-target/gse30219`
  (bản R5, 146 mẫu). Tên `gse30219_r5_fixed` của layout RQ0′ vẫn được nhận (`DATASET_DIR_ALTERNATIVES`).
- `scripts/build_v3_splits.py` (một lệnh cho cả 5×5 và few-shot) và `--selection-root` của `run_e1_classification.py`.

**Những gì v3.1 sửa trong phần của đội code:**
- `build_v3_splits.py`: phần few-shot nay ghi đúng giao thức KBTN Mục 6 qua `survey_v2.fewshot` (5 split × **4 tập con lồng nhau**
  mỗi n_T, cột `subset`). Bản cũ (commit `ad75caa`) chỉ có **một** tập con mỗi n_T (O23, Review 29/10) nên không ước lượng được
  phương sai giữa tập con. Tệp `fewshot_splits.csv` sinh trước 29/10 **phải sinh lại** (`--force`); tệp 5×5 `repeated_splits.csv`
  không đổi (cùng generator, cùng seed), chỉ cần kiểm tra lại SHA-256.
- `load_dataset()` ghi `n_per_class` vào notes và, khi `KLTN_STRICT_N=1` (runner Kaggle đặt sẵn), **dừng** nếu số mẫu mỗi lớp khác
  `config.DATASET_EXPECTED_N` (517/501, 40/18, 85/61, 106/66, 133/43).
- Notebook `kaggle_e1_feature_selection_v3_runner.ipynb`: không truyền `--k` nữa (K_GRID lấy từ config — M18), thêm nhóm
  `RUN_CONSTRAINED` (abess, stability_selection, ktsp) và `RUN_PANELS`, wrapper chạy qua `run_e1_wrappers_all_cohorts.py`,
  cổng 3 kiểm tra `traceable == true` và `commit_hash == HEAD` trong từng `config.json`.

**Áp gói (10 phút).** Trên nhánh `survey-v2-code` tại `2508a2a` (nếu đã có commit sau `2508a2a`, báo GVHD trước khi áp):
```bash
git checkout survey-v2-code && git pull
git am --3way survey-v3.1-code_<hash>.patch        # patch đã được tạo từ 2508a2a; nếu xung đột: git am --abort, báo GVHD
python -m pytest tests -q                            # 89 passed (1 skipped) — không có FAILED
python scripts/smoke_test_survey_v2.py --n-jobs 4    # ≈ 10 phút, dữ liệu giả, kết thúc bằng "Sanity: ĐẠT"
git push                                              # mọi kết quả chỉ được trích khi chạy từ commit đã push
```
Commit bằng danh tính UIT của sinh viên (`git config user.name/user.email` trước khi `git am`; `git am` giữ tác giả của patch,
nên nếu muốn một tác giả duy nhất dùng `git am --3way --no-gpg-sign` rồi `git commit --amend --reset-author`).

**Quy ước mới (Review 29/10):**
- **QĐ18 — E2/E3 trên Kaggle có hiệu chuẩn.** Chạy chính thức bằng `scripts/run_survey_e2_e3_timing.py` đã commit (không dùng
  notebook tự viết). Script tự đo 3 phương pháp mốc (`CALIBRATION_METHODS` = l1_lr, svm_rfe, rf_importance; ≈ 2,5 phút) ở điểm E2
  (n = 814, p = 5.000, k = 10) và ghi `calibration.csv`; phiên thứ hai trở đi chạy với `--calibration-reference <calibration.csv của
  phiên đầu>`: script DỪNG nếu một mốc lệch ≥ 10 % (`CALIBRATION_MAX_DRIFT`); `calibration_check.json` và `config.json` ghi độ lệch.
  Thêm `--isolate-memory` để bộ nhớ đỉnh đo trong tiến trình con mới cho từng (phương pháp, k) (cột `maxrss_isolated_mb`, Y19).
  Phiên đầu (tham chiếu) và mọi phiên sau phải cùng loại máy Kaggle (CPU 4 nhân, BLAS 1 luồng); `machine_config.json` ghi CPU/RAM từng phiên.
  Lệnh chuẩn: `python scripts/run_survey_e2_e3_timing.py --wrappers --isolate-memory [--calibration-reference ...]`.
- **QĐ19 — tái dùng.** Chỉ panel pha A trên **TCGA** (10 phương pháp filter/embedded) chạy 26–28/10 được tái dùng, và chỉ khi
  chạy lại **một** lần lặp (repeat 0) với v3.1 cho danh sách gen **giống hệt** (`scripts/compare_panels.py`, Phụ lục A Review 29/10).
  Mọi thứ khác (GEO pha A + B, pha B TCGA, B0/random-k, E1b, E2/E3, E4) chạy lại bằng script v3.1. Kết quả wrapper 5 cohort × 25 fold
  × 8 k và E4 trên 5 cohort (ngoài phạm vi QĐ16) không dùng.
- **M18 — logic thí nghiệm chỉ nằm trong script đã commit**; notebook chỉ gọi script. `traceable` được tính từ provenance, không gán tay.
- **M19 — kết quả 26–28/10 chỉ dùng để ước lượng chi phí**, không trích vào bài.
- Hạn: đợt 1 **03/11** (E0, E1 không wrapper, E2); được dời sang 04/11 nếu báo GVHD trước 02/11. Đợt 2: 06/11 (E1b, E3, E4+E5), 08/11 (wrapper, E7).

---

## 0. Tóm tắt một trang

**Điều gì đổi so với bản 25/10?** Bài survey chuyển sang **cùng bài toán với khoá luận**: LUAD (0) vs LUSC (1) — nguồn TCGA
(RNA-seq, `tcga_luad_lusc_primary_tumor`) và 4 cohort GEO đích (`gse10245`, `gse30219` = `gse30219_r5_fixed`, `gse37745`,
`gse42127`). Bài toán giai đoạn (GSE68465, P0, "tín hiệu yếu") **không còn thuộc survey**. Ngân sách k ∈ {1, 2, 3, 4, 5, 10, 25, 50}.
**E1 của survey ≡ RQ1 của khoá luận**: chạy MỘT lần, một bộ kết quả, một đường mã (hai pha A/B của đội code, giữ nguyên).
Thêm: 3 phương pháp (abess, stability selection, k-TSP), panel IHC-2/IHC-4 và panel công bố, E0 (kiểm tra tín hiệu),
E1b (đường cong học few-shot = nhánh target-only của RQ2), E7 (3 chiến lược chuyển giao đơn giản = baseline D3 của RQ2),
k* gộp theo QĐ11 cho mọi phương pháp. Bộ 11 bộ phân loại của pha B **không dùng để xếp hạng** (chỉ `--models-extended`, phụ lục).

**Đội code phải giao gì?** Kết quả của 10 việc, mỗi việc là một thư mục có `config.json` (commit hash, SHA-256 dữ liệu và bản chia):

| # | Việc | Script | Trả lời | Hạn (Review 26/10) |
|---|---|---|---|---|
| 1 | Cổng 1′: data card 5 bộ, nhãn kiểm tra chéo, mẫu trùng, G | (tay + `rebuild_gse30219_r5.py`, đoạn code J) | điều kiện 2, 3 | 28/10 |
| 2 | Bản chia 5×5 (5 bộ) và few-shot (4 cohort) | `build_e1_repeated_splits.py`, `build_fewshot_splits.py` | giao thức | 28/10 |
| 3 | Sanity Y10 | `run_sanity_auroc.py` | pipeline có hỏng không | 28/10 |
| 4 | E0 kiểm tra tín hiệu (5 bộ) | `run_survey_e0_signal_check.py` | bước 1 khung quyết định | 03/11 |
| 5 | E1 pha A + pha B (11 phương pháp + panel; B0; random-k; 5 bộ) | `run_e1_feature_selection.py`, `run_e1_classification.py --secondary` | RQ1 (= H1 khoá luận) | 03/11 (bàn giao 1), 06/11 |
| 6 | E1 wrapper (GA, BPSO; 3 cohort GPL570; 3×5; k ∈ {5, 10, 50}) | `run_e1_wrappers_all_cohorts.py` rồi pha B `--methods ga bpso` | RQ1 cho wrapper | 08/11 |
| 7 | E1b đường cong học (4 cohort) | `run_survey_e1b_learning_curve.py` | RQ4 (và ổn định few-shot) | 06/11 |
| 8 | E2 + E3 chi phí (TCGA, máy cố định) | `run_survey_e2_e3_timing.py --wrappers` | RQ2 | 03/11 (E2), 06/11 (E3) |
| 9 | E4 + E5 ổn định (TCGA, GSE37745) | `run_survey_e4_e5_stability.py` | RQ3 | 06/11 |
| 10 | E7 chuyển giao đơn giản (ưu tiên 2) | `run_survey_e7_transfer_naive.py` | RQ4 | 08/11 (nếu không kịp: báo) |
| — | Bảng, hình, manifest (E6 ghép) | `make_survey_tables.py` | tất cả | sau mỗi đợt |

**Ba quy tắc không được vi phạm**

1. *Không chạm vào test*: mọi bước (lọc phương sai, chọn gen, chuẩn hoá, chọn C, chọn ngưỡng, fitness của wrapper) chỉ dùng
   phần train của fold hoặc L_T. Pha B **không** được chọn "best classifier" theo AUROC test (O19) — code đã bỏ tệp đó.
2. *Không sửa hằng số đặt trước* trong `survey_v2/config.py` (k, n_T, seed, M, số hoán vị, lưới C, danh sách cohort cho wrapper…)
   nếu chưa được GVHD duyệt bằng văn bản. Nếu buộc phải sửa để chạy nổi, ghi lý do vào commit message và báo ngay.
3. *Mọi kết quả phải truy vết*: script chạy phải đã `git commit` + `git push`; `config.json` phải có `commit_hash` và
   `traceable = true`. Kết quả chạy với `KLTN_ALLOW_UNPUSHED=1` chỉ để thử, **không gửi**. Commit bằng danh tính sinh viên
   (tên, email UIT) — không dùng danh tính khác (Y17).

---

## 1. Bối cảnh: vì sao đổi và đổi cái gì

Review 26/10 (QĐ15): survey thống nhất với đề tài khoá luận *"Giảm chiều thông qua chọn lọc gen có ràng buộc ngân sách kết hợp
học chuyển giao để phân loại dữ liệu biểu hiện gen nhiều chiều, ít mẫu trong chẩn đoán bệnh"*. Hệ quả cho code:

| Thành phần | Bản 25/10 | Bản v3 (26/10) |
|---|---|---|
| Dữ liệu | 3 bộ giai đoạn (perm/) | TCGA LUAD+LUSC (perm/) + 4 cohort GEO (geo-target/, định dạng của RQ0′) |
| Không gian gen | toàn bộ | cohort GEO: giao với TCGA (loader tự làm; `config.json` ghi `gene_space`) |
| k | {10, 50, 100, 200, 500} | {1, 2, 3, 4, 5, 10, 25, 50}; few-shot {2, 3, 4, 5, 10}; wrapper {5, 10, 50} |
| Phương pháp | 10 | 13 (+ abess, stability selection, k-TSP) + panel IHC-2/4 + panel công bố (CSV) |
| Bộ phân loại | LR-L2; RF, SVM ở k ∈ {50, 200} | LR-L2; RF, SVM ở k ∈ {5, 25}; thêm log-loss, Brier; M5 cho n_T ≤ 10 |
| Mới | — | E0, E1b, E7, k* gộp (QĐ11), few-shot splits |
| Bỏ | P0, bài toán giai đoạn, bộ 11 mô hình để xếp hạng, CodeCarbon mặc định | — |

Bảy điều kiện để kết quả được dùng (Review 26/10, `05_decisions_and_design.md` Mục 3) nằm ở từng script: chia theo bệnh nhân,
lọc không nhãn trong fold, G = giao với TCGA, nhãn kiểm tra chéo, CV lặp + Nadeau–Bengio + k* gộp, máy cố định cho E2–E3,
Nogueira có CI trên M = 30 tập con độc lập, truy vết commit.

---

## 2. Cài đặt và kiểm tra môi trường

```bash
git clone https://github.com/datachain-uit/RNA-Seq-FS-Survey.git
cd RNA-Seq-FS-Survey
git checkout survey-v2-code                               # áp patch v3 lên đây (xem README-GOI-CODE)
python -m venv .venv && source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install abess                                          # L0 (tuỳ chọn; nếu không cài được trên Python 3.14: ghi vào HANDOVER, method abess bị bỏ qua tự động)
export KLTN_DATA_ROOT=/path/to/KLTN                        # Windows PowerShell: $env:KLTN_DATA_ROOT = "D:\KLTN"

python -m pytest tests -q -m "not slow"                   # ~15 s: 75 test đơn vị
python -m pytest tests -q                                 # ~4 phút: thêm 7 test đầu-cuối trên dữ liệu giả
python scripts/smoke_test_survey_v2.py --n-jobs 4          # ~10–20 phút: toàn bộ chuỗi E0–E7 trên dữ liệu giả
```

Dữ liệu cần có trong `$KLTN_DATA_ROOT/`:

```
experiment-data-perm/tcga_luad_lusc_primary_tumor/   expression.h5 (key "expression"), metadata.csv (index sample_id; patient_id, class, label)
experiment-data-perm/tcga_normal_vs_tumor_sanity/    như trên + perm_folds.csv, folds/ (Y10)
experiment-data-geo-target/gse10245/                 expression.h5, metadata.csv (cột sample_id, label, class[, patient_id])
experiment-data-geo-target/gse30219_r5_fixed/        (bản R5 chuẩn, một bản duy nhất — Y12)
experiment-data-geo-target/gse37745/
experiment-data-geo-target/gse42127/
```

Nếu `metadata.csv` của cohort GEO không có `patient_id`, loader đặt patient_id = sample_id và ghi
`patient_id_assumed_from_sample_id = true` vào `config.json` — **chỉ chấp nhận khi kiểm tra mẫu trùng (O15) đã làm và ghi vào data card**.

Kiểm tra nhanh cấu hình máy chạy E2–E3: `python -c "from survey_v2.timing import machine_info; print(machine_info())"`.

---

## 3. Quy tắc truy vết (provenance) — không đổi

`survey_v2.runlog.start_run(__file__, params)` ở đầu mỗi script: dừng nếu kho không sạch, script chưa được theo dõi hoặc HEAD chưa
push (Y13). `config.json` có `commit_hash`, `dirty = false`, `traceable = true`, SHA-256 của dữ liệu **và của tệp bản chia**, phiên bản
thư viện, tham số dòng lệnh. Trên Kaggle: clone đúng nhánh đã push (runner của đội code đã làm đúng).

---

## 4. Cấu trúc code và luồng dữ liệu (v3)

```
survey_v2/
  config.py      registry 5 bộ (DATASETS, DATASET_DIRS), k, n_T, 13 phương pháp, panel IHC, E0–E7, thống kê (k*: KSTAR_*)
  data.py        loader hai layout; không gian gen G = giao với TCGA; CV 5×5 theo bệnh nhân; lọc phương sai; tập con; bản sao
  methods.py     13 phương pháp + random-k: f_test, mutual_info, mrmr, relieff; l1_lr, elastic_net, rf_importance, svm_rfe (tới k = 1);
                 abess (L0), stability_selection, ktsp (k chẵn); ga, bpso
  panels.py      IHC-2/IHC-4 + panel công bố (panels/published_panels.csv — ĐỘI CODE ĐIỀN Girard 2016, Wilkerson 2013)
  classify.py    LR-L2 (chính), RF, SVM tuyến tính; C và ngưỡng bằng CV nội bộ; log-loss, Brier; M5 (n_T ≤ 10: C = 1, ngưỡng 0,5)
  fewshot.py     bản chia few-shot (5 split × 4 tập con lồng nhau × n_T) + kiểm tra
  transfer.py    φ chuẩn hoá mức panel; source_only / select_on_source / pooled / target_only (E7)
  kstar.py       k* gộp QĐ11 (đoạn code G) + k* từng cohort
  e1_splits.py, e1_classification.py (đội code; bộ 11 mô hình chỉ cho --models-extended)
  stats.py, stability.py, timing.py, wrappers.py, mrmr.py, runlog.py, synthetic.py (dữ liệu giả hai layout)
scripts/
  build_e1_repeated_splits.py, build_fewshot_splits.py
  run_survey_e0_signal_check.py
  run_e1_feature_selection.py (pha A), run_e1_wrappers_all_cohorts.py (pha A wrapper), run_e1_classification.py (pha B)
  run_survey_e1b_learning_curve.py, run_survey_e7_transfer_naive.py
  run_survey_e2_e3_timing.py, run_survey_e4_e5_stability.py, make_survey_tables.py, run_sanity_auroc.py, smoke_test_survey_v2.py
  run_survey_e1.py, run_survey_e1_wrappers.py  (ĐÃ THAY THẾ: chỉ in hướng dẫn rồi dừng)
tests/   test_survey_v3_units.py (15), test_survey_v2_scripts_e2e.py (7, slow), + test của bản v2 và của đội code
```

Luồng E1 (= RQ1):

```
repeated_splits.csv (5×5 theo bệnh nhân, seed 42–46; kiểm tra không trùng bệnh nhân)
   │
   ├─ PHA A (run_e1_feature_selection.py): trên outer-train của fold
   │     lọc phương sai KHÔNG nhãn → p0 = 5.000 gen (trong G)
   │     ├─ xếp hạng (f_test, mutual_info, mrmr, relieff, rf_importance, svm_rfe, stability_selection): một lần, tiền tố theo k
   │     ├─ đúng k (l1_lr, elastic_net, abess): dò C / support_size = k; bão hoà → bổ sung F-test, ghi n_filled
   │     ├─ ktsp: cặp gen rời nhau; chỉ k chẵn (k lẻ: n_selected = 0, không có gen)
   │     ├─ ga, bpso (run_e1_wrappers_all_cohorts.py): đúng k, fitness CV 3 fold LR-L2 trong train; 3 cohort GPL570, repeats 0–2, k ∈ {5,10,50}
   │     └─ ihc_2, ihc_4, panel_<tên>: cố định, không phụ thuộc dữ liệu; gen thiếu ghi vào missing_genes
   │     → selection_per_fold.csv, selected_genes.csv, selection_stability.csv (Jaccard mô tả), checkpoints/
   │
   └─ PHA B (run_e1_classification.py): trên cùng fold
         b0 = LR-L2 trên TOÀN BỘ G; random_k = 100 lần rút trong p0 (seed [cohort, seed, fold, k, draw])
         mọi panel pha A → LR-L2 (C bằng CV nội bộ 3 fold theo AUROC; ngưỡng BAcc trên OOF; --secondary: RF, SVM ở k ∈ {5, 25})
         → classification_per_fold.csv (auroc, bacc, mcc, logloss, brier, C), classification_summary.csv, e1_random_k.csv
```

---

## 5. Giao thức chung — "vì sao" cho từng lựa chọn

| Lựa chọn | Làm gì trong code | Vì sao |
|---|---|---|
| G = gen cohort ∩ gen TCGA | `data.load_dataset(gene_space="reference")` | Định nghĩa bài toán (Review 11/10 Mục 8.2.1): panel phải đo được ở cả hai nền tảng; B8 checklist. |
| Lọc sơ bộ 5.000 gen phương sai lớn nhất, không nhãn, trong fold (GVHD chốt 26/10, KBTN M14; thay F-test 2.000 gen của KBTN v2) | `data.variance_prefilter` | Không "F-test hoá" mọi phương pháp; không rò rỉ (Ambroise & McLachlan 2002). [Giả định, chờ GVHD chốt câu 2] |
| CV 5×5 tính trước, seed 42–46 | `e1_splits.py` (đội code) | Trùng RQ0′; mọi phương pháp cùng lần chia → so sánh cặp; wrapper 3 lần lặp đầu. |
| Few-shot: 5 split test_T/pool_T 1/3:2/3; 4 tập con lồng nhau; n_T ∈ {5,10,20,40,pool} mỗi lớp | `fewshot.py` | KBTN v2 Mục 6 (RQ2); lồng nhau để đường cong học ghép cặp theo n_T; GSE10245 chỉ {5, 10}. |
| M5: n_T ≤ 10 → C = 1, ngưỡng 0,5; n_T ≥ 20 → CV nội bộ | `classify.evaluate_panel(small_n=...)` | CV 3 fold trong 10 mẫu không khả thi (Review 24/10 M5). |
| random-k, B0 trong pha B | `run_e1_classification.py` | Mốc sàn/trần phải cùng bộ phân loại, cùng fold với các phương pháp (bản của đội code để chúng ở script cũ). |
| IHC-2/IHC-4 là mốc tri thức | `panels.py` | QĐ2, QĐ12: so "không thua kém", không phải baseline phải vượt. |
| k* gộp (IVW + bằng nhau, chuỗi 50→1) cho mọi phương pháp | `kstar.py` | QĐ8 + QĐ11; survey báo k* như một chỉ số của phương pháp. |
| Không chọn "best classifier" theo test | pha B | O19: chọn trên test là R3 tái diễn. |
| Wrapper chỉ 3 cohort GPL570, 3×5, k ∈ {5,10,50} | `run_e1_wrappers_all_cohorts.py` | Chi phí (O20); vùng chính k ≤ 10 + một điểm k = 50. |
| CodeCarbon tắt mặc định | `--track-emissions` | Y16: Kaggle không phải máy cố định; năng lượng chỉ là ước lượng, nếu báo thì ghi "ước lượng". |

---

## 6. Mười ba phương pháp — điểm mới so với bản 25/10

Giữ nguyên mục 6 của bản 25/10 cho 10 phương pháp cũ (F-test, MI, mRMR, ReliefF, L1-LR, Elastic Net, RF importance, SVM-RFE, GA, BPSO) với hai thay đổi:
SVM-RFE loại dần tới k_min = 1 (để có k = 1, 2, 3); mRMR chỉ xếp tới k_max = 50 (rẻ hơn nhiều).

11. **abess (L0, Zhu et al. 2020, 2022)** — `select_abess`: `abess.LogisticRegression(support_size=k)` trên dữ liệu z-score; đúng k hệ số khác 0
    (nếu thừa: lấy |β| lớn nhất; thiếu: bổ sung F-test, ghi `n_filled`). Nếu `import abess` lỗi: phương pháp bị bỏ qua, ghi `methods_skipped`
    trong `config.json` — **ghi vào HANDOVER**.
12. **Stability selection (Meinshausen & Bühlmann 2010)** — `rank_stability_selection`: chọn C trên toàn bộ train sao cho ≈ 2·k_max hệ số
    khác 0 (q), rồi 100 tập con phân tầng 50 % → tần suất π_j; xếp hạng theo π_j (hoà: |β| trung bình); tiền tố theo k. Ghi `C`, `q_target`.
13. **k-TSP (Tan et al. 2005)** — `rank_ktsp`: điểm cặp Δ_ij = |P(x_i < x_j | 0) − P(x_i < x_j | 1)| tính chính xác cho mọi cặp trong p0 gen
    (O(n p0²), theo khối; `KTSP_MAX_P = 5000`), cặp rời nhau, tham lam; panel k = 2m gen = m cặp đầu → **chỉ k chẵn**; k lẻ ghi
    `n_selected = 0`. Bộ phân loại chính là LR-L2 trên gen của panel (để so cùng bộ phân loại); luật bỏ phiếu k-TSP (`ktsp_vote_scores`)
    là phụ, bổ sung sau nếu kịp.

**Panel cố định**: `ihc_2 = (NKX2-1, TP63)`, `ihc_4 = + (NAPSA, KRT5)`; panel công bố đọc từ `survey_v2/panels/published_panels.csv`
(cột `panel, gene, source`) — **đội code điền từ bài gốc** (Girard et al. 2016 phần AC vs SCC; Wilkerson et al. 2013), ghi DOI.
Gen thiếu trên nền tảng → `missing_genes` trong `selection_per_fold.csv` và `exact_k = False`.

---

## 7. Từng thí nghiệm: mục đích, lệnh, đầu ra, tiêu chí chấp nhận

Mọi lệnh chạy từ thư mục gốc kho, sau khi `export KLTN_DATA_ROOT=...`. Kết quả vào `$KLTN_DATA_ROOT/results/survey_v2/`
(tên thư mục gốc giữ nguyên để runner Kaggle không đổi). Các script **tự tiếp tục** (resume) khi chạy lại.

### 7.0 Cổng 1′ và bản chia (việc #1, #2)

```bash
python scripts/build_e1_repeated_splits.py            # <bộ>/repeated_splits.csv cho 5 bộ + e1_repeated_splits_manifest.json
python scripts/build_fewshot_splits.py                 # <cohort>/fewshot_splits.csv cho 4 cohort + fewshot_splits_manifest.json
# hoặc một lệnh (v3.1, cùng generator và seed, cùng kết quả): python scripts/build_v3_splits.py --force   # --force: sinh lại few-shot cũ (1 tập con)
```
- *Chấp nhận:* 25 (repeat, fold) mỗi bộ, không trùng bệnh nhân, mỗi mẫu test đúng một lần/lần lặp; few-shot: test_T cố định trong split,
  tập con lồng nhau theo n_T, đúng n_T mẫu mỗi lớp; GSE10245 chỉ n_T ∈ {5, 10, pool}. SHA-256 của hai tệp ghi vào manifest và data card.
- Cổng 1′ còn: nhãn 4 cohort kiểm tra chéo theo tập mẫu (đoạn code J), mẫu trùng (đoạn code 7.2.2 Review 20/10), GSE43580 cách ly,
  data card đủ trường, số gen G mỗi cohort (lấy từ `config.json` của pha A: `gene_space`).

### 7.1 Sanity Y10 — như bản 25/10 (`python scripts/run_sanity_auroc.py`; AUROC ≥ 0,99 cho 3 pipeline ở k = 10).

### 7.2 E0 — kiểm tra tín hiệu (việc #4)

- *Chứng minh:* mỗi cohort có tín hiệu LUAD/LUSC thật (hoán vị) và chọn gen hơn random-k; thủ tục cho bước 1 của Fig. 9.
- *Lệnh:* `python scripts/run_survey_e0_signal_check.py --n-jobs 4` (200 hoán vị + 100 random-k, k = 10, F-test → LR-L2, CV 5 fold).
- *Đầu ra:* `E0/<bộ>/{e0_signal_check.csv, e0_perm_scores.csv, e0_random_k_scores.csv, config.json}`.
- *Chấp nhận:* `n_perm = 200`, `n_random = 100`; `p_perm` ≥ 1/201; kỳ vọng `signal_present = True` và `selection_useful = True` ở cả 5 bộ.
  Nếu một cohort không đạt: KHÔNG sửa ngưỡng, báo GVHD (có thể là lỗi nhãn).
- *Thời gian:* < 1 giờ.

### 7.3 E1 — pha A, wrapper, pha B (việc #5, #6)

Ablation loại 4 gen IHC (ưu tiên 2, thuộc khoá luận): thêm `--exclude-genes NKX2-1 TP63 NAPSA KRT5` cho cả pha A (với `--out` riêng)
và pha B (`--selection-roots` trỏ tới thư mục đó, `--out` riêng); B0 và random-k khi đó cũng không chứa 4 gen này.

```bash
python scripts/run_e1_feature_selection.py                                   # 11 phương pháp + ihc_2, ihc_4 (+ panel_*), 5 bộ, 8 k
python scripts/run_e1_wrappers_all_cohorts.py                                # GA + BPSO; 3 cohort GPL570; repeats 0 1 2; k 5 10 50
python scripts/run_e1_classification.py --secondary                          # LR-L2 (+ RF, SVM ở k ∈ {5, 25}); B0; random-k; gộp panel wrapper
```
Có thể chia nhỏ theo `--datasets`, `--methods`, `--repeats`, `--k`; Kaggle: một Save Version = một bộ (pha A) — xem notebook
`kaggle_e1_separated_5x5_runner.ipynb` (đã cập nhật COHORTS, K_GRID, wrapper) hoặc `kaggle_e1_feature_selection_v3_runner.ipynb` (v3.1).

*Kiểm tra bằng nhau để tái dùng panel TCGA (QĐ19, làm trước khi chạy pha A TCGA đầy đủ):*
```bash
python scripts/run_e1_feature_selection.py --datasets tcga_luad_lusc_primary_tumor --repeats 0 \
    --methods f_test mutual_info mrmr relieff l1_lr elastic_net rf_importance svm_rfe --out E1_feature_selection_eqcheck
python scripts/compare_panels.py --old <thư mục E1_feature_selection chạy 26–28/10> \
    --new $KLTN_DATA_ROOT/results/survey_v2/E1_feature_selection_eqcheck --repeats 0
```
`panel_equality_summary.csv`: phương pháp có `verdict = REUSE` (mọi fold × k của repeat 0 giống hệt) được chép panel cũ của TCGA vào
`E1_feature_selection/tcga_luad_lusc_primary_tumor/` (kèm `config.json` cũ và ghi chú trong HANDOVER); phương pháp `RERUN` chạy lại
đủ 5 lần lặp bằng v3.1. GEO: luôn chạy lại (gene space G và bản chia few-shot đã đổi).

*Cặp mẫu nghi trùng (Y22 → QĐ21):* `python scripts/check_duplicate_pair.py --dataset gse37745 --samples GSM1019246 GSM1019247` — kết quả 29/10:
cả hai là nữ theo biểu hiện, GSM1019247 ghi nam → cách ly cả hai (GSE37745 còn 170 = 105/65).

*Giới tính theo biểu hiện cho MỌI mẫu (QĐ22):* `python scripts/check_sex_consistency.py` (5 bộ; cần cột sex/gender trong metadata.csv lấy từ
series matrix; GSE10245 không có → chỉ phân loại). Hai trục XIST và trung bình gen Y được chia hai cụm k-means trên chính cohort (tự hiệu chuẩn
theo nền tảng). Cách ly: `opposite` (cả hai trục trái metadata) và `inconsistent` (XIST và gen Y cùng cao); giữ và ghi cờ: `inconclusive`
(both-low, hoặc chỉ một trục trái, hoặc trục không phân biệt). Không sửa nhãn. Đầu ra `qc/sex_check_<bộ>.csv`, `qc/sex_check_summary.csv`
nộp cùng Cổng 1′; mẫu cách ly ghi vào `excluded_samples_<bộ>.csv`, cập nhật `DATASET_EXPECTED_N` (một commit "QĐ22") rồi mới sinh bản chia.
TCGA v2 chỉ có gen mã hoá protein (không có XIST): script tự chuyển sang chế độ MỘT trục gen Y **bỏ phiếu theo từng gen** (QĐ22b, bản v3.1.3):
ngưỡng từng gen = trung điểm (phân vị 95 nhóm ghi nữ, phân vị 5 nhóm ghi nam); M-like(Y) khi ≥ 4/5 gen vượt ngưỡng, F-like(Y) khi ≤ 1/5,
còn lại intermediate(Y) → `inconclusive` (giữ, ghi cờ); `opposite` → cách ly. Không dùng trục trung bình gen Y với biên: u của nam thường
giảm biểu hiện Y (mất một phần nhiễm sắc thể Y) nên trục trung bình cho 5 "opposite" giả ở nam LUSC (29/10). Cột sex của TCGA lấy từ GDC
(`sex_at_birth`) và ghi vào `metadata.csv` bằng script đã commit.
Kết quả 29/10 (sau QĐ21 + QĐ22 + QĐ22b): GSE10245 58 → 55 (37/18), GSE30219 146 → 141 (81/60), GSE37745 172 → 169 (104/65), GSE42127 176 → 167
(126/41); TCGA 1.018 → 1.016 (515/501): cách ly TCGA-49-4505-01A, TCGA-49-AARN-01A (5/5 gen Y ở mức nam, ghi nữ); 5 mẫu `inconclusive` giữ và
ghi cờ (TCGA-21-5782-01A; TCGA-33-4582-01A, TCGA-85-A4QR-01A, TCGA-MF-A522-01A, TCGA-NC-A5HG-01A). Sàng r_z ≥ 0,90: không có cặp. Vì bản chia
TCGA đổi, điều khoản tái dùng panel của QĐ19 không còn: pha A TCGA chạy lại đầy đủ bằng v3.1.

- *Đầu ra:* `E1_feature_selection/<bộ>/{selection_per_fold.csv, selected_genes.csv, selection_stability.csv, selection_gene_frequency.csv, config.json}`;
  `E1_wrapper_feature_selection/<cohort GPL570>/...`; `E1_classification/<bộ>/{classification_per_fold.csv, classification_summary.csv, e1_random_k.csv, config.json}`.
- *Chấp nhận:*
  - pha A: mỗi (repeat, fold, method, k) đúng 1 dòng; `n_selected == k` trừ ktsp ở k lẻ (= 0) và panel cố định thiếu gen; `n_filled > 0` chỉ ở
    l1_lr / elastic_net / abess khi bão hoà (ghi nhận, không xoá); `config.json`: `traceable = true`, `prefilter_p0 = 5000`, `gene_space` ghi
    `intersection with tcga_luad_lusc_primary_tumor` ở 4 cohort GEO, `methods_skipped` rỗng (hoặc chỉ `abess` kèm lý do trong HANDOVER).
  - wrapper: 15 fold × 3 k × 2 wrapper × 3 cohort; `n_eval > 0`.
  - pha B: mỗi (repeat, fold, method, k) có đúng 1 dòng `lr_l2` (+ `rf`, `svm_linear` ở k ∈ {5, 25}); có `b0` (k = |G|) và `random_k`
    (100 lần/fold trong `e1_random_k.csv`); `brier`, `logloss` không NaN ở lr_l2/rf; **không có** tệp `best_classifier_by_panel.csv`.
- *Thời gian ước lượng* (một luồng/fold; GEO n ≈ 120–140, TCGA n ≈ 814; p0 = 5.000): pha A GEO ≈ 5–12 phút/fold (Elastic Net và stability
  selection đắt nhất; k-TSP ≈ 1 phút), TCGA ≈ 15–30 phút/fold → 125 fold ≈ 25–40 giờ CPU, chia 4–5 tiến trình ≈ 6–10 giờ. Wrapper: 15 fold ×
  3 k × 2 × 3 cohort × 5–10 phút ≈ 20–45 giờ CPU. Pha B: RF 500 cây ở k ∈ {5, 25} chiếm phần lớn, ≈ 1 giờ/bộ.

### 7.4 E1b — đường cong học few-shot (việc #7)

- *Chứng minh (RQ4, RQ3):* các phương pháp hiện có suy giảm thế nào khi n_T ∈ {5, 10, 20, 40} mẫu/lớp dưới ngân sách k ≤ 10; n_T*; ổn định few-shot.
- *Lệnh:* `python scripts/run_survey_e1b_learning_curve.py --n-jobs 4` (4 cohort; 11 phương pháp + ihc_2, ihc_4 + random-k; k ∈ {2, 3, 4, 5, 10}).
- *Đầu ra:* `E1b/<cohort>/{e1b_per_run.csv, e1b_selected_genes.csv, e1b_random_k.csv, e1b_stability.csv, config.json}`.
- *Chấp nhận:* 20 lần chạy (5 split × 4 subset) ở mỗi mức n_T có trong bản chia; `C = 1` ở n_T ≤ 10 (M5); `fewshot_splits_sha256` trong config
  trùng manifest; `n_runs = 20` trong `e1b_stability.csv`.
- *Thời gian:* mỗi lần chạy (một tập con) ≈ 1–3 phút cho 11 phương pháp → 4 cohort × 20 × 4–5 mức ≈ 8–12 giờ CPU.

### 7.5 E7 — chuyển giao đơn giản (việc #10, ưu tiên 2; GVHD chốt 26/10 giữ E7 trong survey, hạn cắt 10/11)

- *Chứng minh (RQ4):* với n_T ∈ {5, 10, 20}, source_only (zero-shot, φ mức panel), select_on_source, pooled so với target_only trên cùng tập con;
  chuyển giao âm = CI của hiệu nằm dưới 0.
- *Lệnh:* `python scripts/run_survey_e7_transfer_naive.py --n-jobs 4` (sau E1b; cùng `fewshot_splits.csv`).
- *Đầu ra:* `E7/<cohort>/{e7_per_run.csv, e7_selected_genes.csv, config.json}` (config ghi panel nguồn theo k và |G|).
- *Chấp nhận:* 4 chiến lược × 5 k × 20 lần chạy × 3 mức n_T mỗi cohort; `source_only` cùng giá trị cho mọi tập con trong một split.
- *Thời gian:* ≤ 6–10 giờ CPU (pooled fit L1 trên ≈ 1.000 + 2 n_T mẫu × 5.000 gen).

### 7.6 E2 + E3 — chi phí (việc #8): như bản 25/10, trên TCGA LUAD+LUSC, `--k 10 50`, lưới n {100, 200, 400, all}; k-TSP ghi NaN khi p > 5.000;
BLAS 1 luồng, `machine_config.json`, `wall_spread_max_rel < 0,3`. v3.1 (QĐ18): được chạy trên Kaggle **có hiệu chuẩn** — phiên tham chiếu
chạy trước, các phiên sau dùng `--calibration-reference`; `config.json` phải có `calibration_ok = true` (hoặc `null` ở phiên tham chiếu);
`--isolate-memory` bắt buộc cho E2 (bộ nhớ đỉnh = `maxrss_isolated_mb`). Số đo 26–28/10 (notebook, 6 phiên, không hiệu chuẩn) chỉ để ước lượng
chi phí (M19), không trích.
v3.1.4 (QĐ23, 29/10): E2 chính thức = **phiên tham chiếu** (`e2_ref`: 13 phương pháp + wrapper trong một phiên, ≈ 6–9 giờ); các phiên E3
chia phần chạy `--skip-e2` (chỉ hiệu chuẩn + E3) và được gộp bằng `kaggle_jobs.py merge-e2e3` (E2 từ phiên tham chiếu, E3 từ các phần;
config.json của thư mục gộp stamp bằng provenance). `C.E3_MAX_P = {"elastic_net": 10000}`: Elastic Net (saga) đo tới p = 10.000 vì điểm
p = 19.938 cần ≈ 5 giờ cho 3 lần đo và không vừa một phiên; bài ghi "đo tới p = 10.000", số mũ fit trên 4 điểm.
v3.1.6 (QĐ27, 31/10): `measure()` không còn tracemalloc (wall time của ReliefF/MI/RF/wrapper trong các phiên 29–30/10 bị phóng đại
2,5–5,7 lần); bộ nhớ tiến trình con đo bằng VmHWM (ru_maxrss kế thừa đỉnh của tiến trình cha → các cột `maxrss_*` cũ vô nghĩa);
cột `peak_mb`, `maxrss_mb` bỏ. `merge-e2e3` từ chối phiên có `timing_tracemalloc != false` hoặc `peak_source != "VmHWM"`.
**Mọi phiên E2/E3 chạy lại ở v3.1.6** (ước lượng sau khi bỏ tracemalloc: `e2_ref` ≈ 3 giờ, `e3_p1` ≈ 2 giờ, `e3_p2` ≈ 4–5 giờ (Elastic Net),
`e3_p3` ≈ 1,5–2 giờ). Kết quả 29–30/10 chuyển vào 04_Luu-vet với ghi chú "KHÔNG DÙNG — QĐ27".

### 7.7 E4 + E5 — ổn định (việc #9): như bản 25/10 nhưng `--datasets` mặc định = TCGA, GSE37745; E5a và E5b chỉ GSE37745; E5b gồm
rf_importance, stability_selection (mọi k) và GA, BPSO (k = 10); E5c ở k = 10 từ `selected_genes.csv` của pha A (+ wrapper). Random-k ở sàn
≈ k/5.000; mọi phương pháp cao hơn sàn (CI không chồng lấn) ở ít nhất một k.

### 7.8 Bảng, hình, E6: `python scripts/make_survey_tables.py` → `tables/*.csv|*.tex`, `figures/*.pdf|*.png`, `provenance_manifest.json`.
Mới: `e0_signal_check`, `e1_kstar_pooled(_by_k)`, `e1_kstar_per_cohort`, `e1_bayes_rope`, `e1b_learning_curves(_table)`, `e1b_stability`,
`e7_transfer_table`, `fig_e1b_learning_curve_<cohort>`, `fig_e7_transfer_<cohort>`; E6 tại k = 10.

---

## 8. Thống kê dùng trong bài — bổ sung so với bản 25/10

- **k* gộp (QĐ8 + QĐ11, `survey_v2/kstar.py`):** với mỗi phương pháp, chuỗi k cố định 50 → 25 → 10 → 5 → 4 → 3 → 2 → 1 trên các mức đã
  chạy; mức k "đạt" khi cận dưới một phía 95 % của ΔAUROC(panel − B0) gộp qua 3 cohort GPL570 > −0,02 theo IVW (phương sai Nadeau–Bengio)
  **và** trọng số bằng nhau; dừng ở lần đầu không đạt. Báo `k_star_ivw`, `k_star_both`, trọng số từng cohort, I², `k_sequence_used`.
  Quy tắc "ủng hộ/bác bỏ" chỉ có nghĩa cho H1 của L1-LR; với phương pháp khác là mô tả.
- **Few-shot:** CI bootstrap qua các lần chạy (2.000 lần) cho E1b; hiệu ghép cặp + Wilcoxon + Holm trên 3 mức n_T cho E7.
- **log-loss, Brier (M9):** chỉ khi điểm là xác suất (LR-L2, RF); SVM tuyến tính ghi NaN.
- Người viết chỉ được viết "tốt hơn" khi `p_holm < 0,05` hoặc ngoài ROPE; trên GSE30219/GSE37745 AUROC bão hoà từ k ≈ 2–4 nên so sánh ở
  k ≤ 3, ở GSE10245/GSE42127 và theo log-loss/Brier.

---

## 9. Bàn giao cho người viết

Copy (không đổi tên) các thư mục sau vào `input/results/` của không gian viết bài, kèm `config.json`:

```
input/results/
  data_cards/             DATA_CARD_<5 bộ>.md + DATA_CARD_tcga_normal_vs_tumor_sanity.md
  splits/                 <bộ>/repeated_splits.csv (5), e1_repeated_splits_manifest.json, <cohort>/fewshot_splits.csv (4), fewshot_splits_manifest.json
  sanity/                 sanity_summary.csv, sanity_per_fold.csv, config.json
  E0/<bộ>/                e0_signal_check.csv, e0_perm_scores.csv, e0_random_k_scores.csv, config.json
  E1_feature_selection/<bộ>/          selection_per_fold.csv, selected_genes.csv, selection_stability.csv, config.json
  E1_wrapper_feature_selection/<cohort>/  như trên
  E1_classification/<bộ>/             classification_per_fold.csv, classification_summary.csv, e1_random_k.csv, config.json
  E1b/<cohort>/           e1b_per_run.csv, e1b_selected_genes.csv, e1b_stability.csv, config.json
  E2_E3/                  e2_cost_point.csv, e3_scaling_n.csv, e3_scaling_p.csv, e3_wrapper_evals.csv, e3_exponents.csv, machine_config.json, config.json
  E4_E5/<bộ>/             e4_stability.csv, e5a_redundancy.csv, e5b_seed_stability.csv, e5c_overlap_k10.csv, subsamples.csv, config.json
  E7/<cohort>/            e7_per_run.csv, e7_selected_genes.csv, config.json
  tables/, figures/       toàn bộ (kèm provenance_manifest.json)
  HANDOVER.md             commit hash từng thư mục, ngày chạy, máy (E2–E3), thời gian thực tế, cảnh báo, sai lệch, phương pháp không chạy
```

Checklist trước khi gửi:

- [ ] `python -m pytest tests -q` xanh trên máy chạy (và trên Kaggle: cell test của runner).
- [ ] Mọi `config.json` có `traceable: true`, `dirty: false`, `commit_hash` khớp `HANDOVER.md`; commit mang danh tính sinh viên.
- [ ] Sanity đạt (≥ 0,99); E0 đạt ở 5 bộ.
- [ ] E1 pha A: 25 fold × 8 k đủ cho 11 phương pháp ở 5 bộ; `gene_space` = giao với TCGA ở 4 cohort GEO; wrapper 15 fold × 3 k × 3 cohort.
- [ ] E1 pha B: có `b0`, `random_k` (100 lần/fold), `ihc_2`, `ihc_4`; không có "best classifier by test".
- [ ] E1b: 20 lần chạy mỗi mức n_T; E7 (nếu có): source_only không đổi theo tập con.
- [ ] E2–E3: `machine_config.json` đầy đủ; dao động nhỏ. E4: M = 30; random-k ở sàn.
- [ ] `make_survey_tables.py` chạy không lỗi; `provenance_manifest.json` liệt kê mọi tệp đầu vào.

---

## 10. Lịch (Review 26/10, Bảng lộ trình)

| Ngày | Việc | Ghi chú |
|---|---|---|
| 30–31/10 | Áp patch v3.1, `pytest`, smoke test, push (danh tính UIT); sinh lại few-shot (`build_v3_splits.py --force`); kiểm tra bằng nhau panel TCGA (QĐ19); hoàn tất Cổng 1′ (data card, code J, GSE43580, sanity Y10) | 31/10: `published_panels.csv` |
| 01–03/11 | E0 (5 bộ, v3.1); E1 pha A + B (5 bộ; TCGA pha A tái dùng nếu khớp); E2 trên Kaggle có hiệu chuẩn (QĐ18) | 03/11: bàn giao đợt 1 (E0, E1 không wrapper, E2); dời 04/11 nếu báo trước 02/11 |
| 04–08/11 | E1b (4 tập con); E3; E4 + E5 (TCGA, GSE37745); E1 wrapper (3 cohort GPL570); E7; `make_survey_tables.py`; HANDOVER | 06/11: E1b, E3, E4–E6; 08/11: wrapper, E7 |
| 09–10/11 | Bổ sung theo yêu cầu người viết; ablation loại 4 gen IHC (khoá luận): pha A và B với `--exclude-genes NKX2-1 TP63 NAPSA KRT5 --out E1_feature_selection_noIHC` / `--selection-roots .../E1_feature_selection_noIHC --out E1_classification_noIHC` | 10/11: RQ1 chốt |

Cập nhật 31/10 (Review 31/10, sau đợt 1): đợt 1 đã xong 17 phiên (29–30/10, sớm hơn hạn 03/11). Lịch còn lại:

| Ngày | Việc | Ghi chú |
|---|---|---|
| 02/11 | Áp v3.1.6, `pytest`, `make_kaggle_notebooks.py --check`, push; `published_panels.csv` (Girard 2016, Wilkerson 2013) + test đọc panel; chuyển e2_ref/e3_p2/e3_p3 cũ vào 04_Luu-vet | hạn `published_panels.csv` đã qua (31/10) |
| 03–04/11 | `redo_svm_rfe_A` → `redo_svm_rfe_B`; `e2_ref` mới; `ablation_phaseB` | 04/11: HANDOVER đợt 1 bổ sung (svm_rfe) |
| 05–06/11 | `e3_p1`, `e3_p2`, `e3_p3` (3 phiên song song, cùng loại máy với `e2_ref` mới) → `e2e3_merge`; `tables` | 06/11: bàn giao E2/E3 + bảng chính thức đợt 1–2 |
| 07–08/11 | Dự phòng chạy lại phiên lệch hiệu chuẩn; bổ sung theo yêu cầu người viết | 08/11: đợt 3 (theo kế hoạch cũ đã xong sớm) |

---

## 11. Lỗi thường gặp (bổ sung)

| Hiện tượng | Nguyên nhân | Xử lý |
|---|---|---|
| `không có gen chung với tcga_luad_lusc_primary_tumor` | tên gen cohort GEO không phải gene symbol như TCGA | kiểm tra quy tắc gom probe → gene symbol của notebook dựng cohort; không sửa tay |
| `fewshot_splits.csv không hợp lệ` | bản chia cũ / sửa tay | chạy lại `build_fewshot_splits.py --force` sau khi kiểm tra; không chỉnh tay |
| `abess` không cài được trên Python 3.14 | chưa có wheel | ghi vào HANDOVER; pha A ghi `methods_skipped = ["abess"]`; thử Python 3.12 trên Kaggle nếu muốn có abess |
| k-TSP `KTSP_MAX_P` | p > 5.000 (E3 ở p lớn) | bình thường; E3 ghi NaN cho k-TSP ở p > 5.000 |
| `n_selected = 0` ở ktsp | k lẻ | bình thường (KBTN 5.1); người viết báo cáo "—" |
| E1 pha B chậm | RF 500 cây ở k ∈ {5, 25} | chia theo `--datasets`; không giảm số cây nếu chưa hỏi |
| `source_only` AUROC < 0,5 | lệch nền tảng khiến φ đảo chiều | đây là kết quả (chuyển giao âm), không phải lỗi; báo cáo nguyên |
