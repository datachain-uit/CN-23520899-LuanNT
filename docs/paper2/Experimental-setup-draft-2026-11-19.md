# Experimental setup (draft, 19/11/2026)

Draft of the Experimental setup section of paper 2 (checklist PH-05; package 22/11, QĐ127). It describes data, protocol,
compared methods, evaluation and statistics only. It contains no result sentences: those are written after the advisor's
recomputation from the result zips, per cohort first (QĐ97d), and within the limits of Table 7 of Review 12/11 and QĐ181.
Citations are given as author–year placeholders until the target journal and its template are chosen (PA-06).

---

## 4 Experimental setup

### 4.1 Data

The task is LUAD (0) versus LUSC (1). The source cohort is TCGA (LUAD and LUSC primary tumours, RNA-seq, log2(TPM + 1)). The
targets are four GEO microarray cohorts (Table 4.1): three on Affymetrix GPL570 (GSE10245, GSE30219, GSE37745), which form the
primary pool, and GSE42127 on Illumina GPL6884, which is reported separately as a second platform. The data, quality control and
splits are those of our survey of exact-k gene selection [paper 1], which reports the same cohorts with their data cards.

**Table 4.1.** Cohorts after quality control.

| Cohort | Platform | LUAD / LUSC | Genes; shared with TCGA (\|G\|) | Role |
|---|---|---|---|---|
| TCGA LUAD/LUSC | RNA-seq | 515 / 501 | 19,938 | source |
| GSE10245 | GPL570 | 37 / 18 | 20,848; 16,479 | target, primary pool |
| GSE30219 | GPL570 | 81 / 60 | 20,848; 16,479 | target, primary pool |
| GSE37745 | GPL570 | 104 / 65 | 20,848; 16,479 | target, primary pool |
| GSE42127 | GPL6884 | 126 / 41 | 19,405; 17,235 | target, second platform (reported separately) |

Quality control removed samples whose recorded sex disagreed with sex-chromosome expression and screened for duplicate
samples within and between cohorts (raw correlation ≥ 0.99 or correlation of gene-wise z-scores ≥ 0.90; each flagged pair decided
individually). For each target cohort the gene space G is the intersection of its gene symbols with those of TCGA; all four
immunohistochemistry (IHC) marker genes (NKX2-1, TP63, NAPSA, KRT5) are present on both platforms.

In a sensitivity analysis, the 39 basaloid squamous carcinomas of GSE30219, excluded from the main data because the series
labels them neither adenocarcinoma nor squamous carcinoma, are added to the LUSC class following the WHO 2015 classification
[Travis 2015] (180 samples, 81 LUAD / 99 LUSC after the same quality control). As descriptive support for this mapping only, the
basaloid samples are compared with the centroids of the adenocarcinomas and squamous carcinomas on the 2,000 genes of largest
variance, chosen without labels.

### 4.2 Few-shot protocol

Each target cohort is split five times (seeds 42–46) into a test set T_e (one third, stratified by class) and a pool (two
thirds). From each pool, four nested training subsets are drawn for n_T ∈ {5, 10, 20, 40} samples per class and the full pool
("pool"), giving 20 runs (5 splits × 4 subsets) per level. Smaller subsets are contained in larger ones within a split and
subset index, so differences between levels are not due to different patients. GSE10245 supports n_T ∈ {5, 10} only, and GSE42127
does not support n_T = 40 (41 LUSC samples). The same split files are used for every method and in the survey.

The budget k is the panel size; k ∈ {2, 3, 4, 5, 10}. The primary quantity for every comparison is the difference in AUROC
averaged over k ∈ {2, 3, 5}; k = 2 and k = 4 also correspond to the two IHC panels.

### 4.3 Compared methods

All methods return exactly k genes. Before selection, the 5,000 genes of largest variance are kept without using labels (on the
source for the transfer strategies, on the target training set for target-only). Expression is z-scored within the training data
before selection.

- **Stability-weighted transfer (MVP).** Selection probabilities π_j are estimated once per gene space by stability selection
  [Meinshausen & Bühlmann 2010] with L1-regularised logistic regression on TCGA (100 subsamples of half the samples). The target
  selector is L1-regularised logistic regression on the few target samples with penalty Σ_j w_j |β_j|, w_j = (π_j + 0.05)^(−γ),
  implemented by rescaling the z-scored columns by 1/w_j. The exponent γ is chosen on the target training data among
  {0.5, 1, 2} (primary configuration). With n_T ≤ 10 per class, γ is chosen by the out-of-fold log-loss, preferring the smaller γ
  when the difference is below 0.01; with n_T ≥ 20, by repeated cross-validation (5 folds × 3).
- **Target-only.** The same selector with γ = 0, i.e. unweighted L1-regularised logistic regression on the target data.
- **Naive transfer** [paper 1]: *select-on-source* fixes the panel selected on TCGA; *pooled* selects on the concatenation of the
  source and the target training data, each z-scored separately.
- **Knowledge panels.** IHC-2 (NKX2-1, TP63) at k = 2 and IHC-4 (+ NAPSA, KRT5) at k = 4.
- **Transfer baselines.** ptLasso [Craig et al.], with the source as external data (stage 1 on TCGA, stage 2 on the target with
  the stage-1 linear predictor as offset and a lighter penalty on the stage-1 support), and Trans-GLM [Tian & Feng 2023] with
  automatic source detection. Both are fitted in R and forced to k genes by the largest absolute coefficients. Their results on
  the cells shared with the survey are reported there [paper 1]; here they enter only a descriptive comparison with MVP.

Two further arms isolate the mechanism of MVP: target-only restricted to the source-filtered gene space (γ = 0 with w ≡ 1), and
the original selection rule that also admits γ = 0 ("m5_cv"), reported with each fixed γ as registered ablations.

When an L1 path cannot reach exactly k non-zero coefficients, the panel is completed by the univariate F-test on the training
data; the number of completed genes is recorded for every run.

### 4.4 Evaluation

Each panel is evaluated by the same classifier: L2-regularised logistic regression with balanced class weights on standardised
features. With at most 10 training samples per class, the regularisation strength is fixed (C = 1) and the threshold is 0.5;
otherwise C ∈ {0.01, 0.1, 1, 10} and the threshold are chosen by three-fold cross-validation within the training data. No step
uses the test set.

The primary metric is AUROC on T_e. Balanced accuracy, Matthews correlation, log-loss and Brier score are reported as secondary
metrics. Panel stability across the 20 runs of a cell is measured by the stability index of Nogueira et al. [2018] with d = |G|.

### 4.5 Hypotheses and statistics

The hypotheses and their criteria were registered before the target results were opened.

- **H2.** At n_T ≤ 20, MVP improves AUROC over target-only, and the gain decreases as n_T grows.
- **H2b** (reported only). At n_T ∈ {5, 10}, MVP is not worse than the IHC panel of the same size (margin 0.02).
- **H2c.** MVP is not worse than select-on-source and pooled (margin 0.01; H0: Δ ≤ −0.01).
- **H4a.** MVP raises the Nogueira index by at least 0.1 over target-only at n_T ≤ 20.
- **H4b** (reported only). With randomly permuted source labels, MVP is not worse than target-only (margin 0.01).

For each cohort, comparison and level of n_T, the uncertainty of the paired AUROC difference is estimated by a two-level
bootstrap with 2,000 resamples: the 20 runs are resampled with replacement and, within each resample, the test patients of each
split are resampled once, stratified by class, and shared by all subsets, budgets and arms of that split. Cohorts are combined by
inverse-variance weighting (IVW) and, as a check, with equal weights; between-cohort heterogeneity is reported as I² together with
the cohort weights, and per-cohort results are shown before pooled ones. One-sided tests at α = 0.05 are adjusted by Holm's method
within each family: the three levels n_T ≤ 20 for H2, and six tests for H2c (two strategies × three levels). The Nogueira index of
the two arms is compared with its asymptotic variance; a split-level bootstrap is reported as a sensitivity analysis.

Descriptive analyses (no multiplicity adjustment) complement the hypotheses: a decomposition of the gain of MVP over target-only
into the contribution of the source gene space and that of the weights π; the overlap between the panels of MVP and of naive
transfer; whether the variant admitting γ = 0 falls back to target-only when the source labels are permuted; a negative control
in which naive transfer is run with the same permuted source labels; and the comparison with ptLasso and Trans-GLM. Each analysis
has a reading rule fixed before it was computed, with one exception: the rule that maps the combination of the overlap and
fallback results to the wording of our contribution was added after these two analyses had been computed, because the
pre-specified rules did not cover the observed combination. We report it as such.

### 4.6 Blind order and checks

Every analysis script was committed and pushed before it read the target results, and the time of first opening is recorded with
the results. The target-only, naive-transfer and IHC arms were re-run with the functions and seeds of the survey so that per-sample
scores were available for the paired bootstrap; an equality gate required every re-run row to match the survey on five metrics within
10⁻¹², with no missing, extra or duplicated rows (4,580 of 4,580 rows). A pilot on a pseudo-split of TCGA (70 % pseudo-source, 30 % pseudo-target) was used, before any GEO result,
to correct the grid of the regularisation path for small weights and to fix the γ rule; it is not used as evidence.

### 4.7 Software and reproducibility

Experiments ran on Kaggle (Python 3.13, Linux, 4 CPU cores) with numpy 2.1.3, pandas 2.3.3, scipy 1.16.3 and scikit-learn 1.6.1;
the pipeline stops if a session's versions differ from those of the survey's runs. The R baselines used R 4.5.3, glmnet 5.0,
ptLasso 1.0.2 and glmtrans 2.1.0, with sessionInfo() stored with the results. Every result file records the commit, parameters,
software versions and SHA-256 checksums of its data and splits, and is citable only if the code was committed and pushed. All
tables and figures of Sections 5–6 are regenerated from the result files by one command
(`scripts/make_rq2_rq4_all.py`), which also checks every table against the archived version.

---

## Ghi chú cho GVHD (không thuộc bản thảo)

- Nguồn từng chi tiết: số mẫu sau QC, |G|, vai trò cohort: KBTN v2.7 Mục 3 (Bảng dữ liệu). Few-shot, MVP, baseline R, thống kê,
  phân tích bắt buộc: KBTN v2.7 Mục 6, Mục 9. Bộ phân loại và M5: `survey_v2/classify.py` (`evaluate_panel`, `_fit_predict`),
  `survey_v2/config.py` (`C_GRID`, `SMALL_N_T_MAX`, `INNER_CV_FOLDS`, `STABSEL_N_SUBSAMPLES`, `STABSEL_FRACTION`, `PREFILTER_P0`,
  `FEWSHOT_N_T_LIMIT`). Phiên bản phần mềm và máy: `config.json` của `RQ2_mvp` trong `rq2_mvp_1b7c021_results.zip`
  (Python 3.13.18, Linux, 4 CPU). Cổng bằng nhau 4.580/4.580: checklist PE-04.
- Câu ở cuối Mục 4.5 nói công khai rằng quy tắc QĐ181 được viết sau khi đã thấy kết quả QĐ126 (a)–(c), đúng như KBTN v2.7 ghi.
  Đề nghị GVHD duyệt cách nói này.
- Chưa có: tạp chí và mẫu định dạng (PA-06), nên trích dẫn còn là chỗ giữ tên tác giả, năm; tên các mục 5–6 là giả định.
- Không có câu kết quả nào trong nháp này (PH-05; Bảng 7 Review 12/11).
- Mô tả MVP ở 4.3 đủ cho người đọc phần thiết lập; giả mã, độ phức tạp, lưới C và quy tắc bù F-test thuộc Method (PC-05, PH-04, hạn 29/11).
