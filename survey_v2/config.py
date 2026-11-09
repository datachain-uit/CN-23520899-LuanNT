"""Hằng số đặt trước của bộ thực nghiệm survey v3 (QĐ15, Review 26/10/2026; KBTN v2.1 M10–M13).

Bài toán: LUAD (0) vs LUSC (1) — bài toán chính của khoá luận. Nguồn TCGA (RNA-seq) và bốn cohort GEO đích
(3 GPL570 + GSE42127). E1 của survey ≡ RQ1 của khoá luận (một thí nghiệm, một bộ kết quả).

Mọi thay đổi ở đây phải được ghi lý do và duyệt trước khi chạy (nguyên tắc đặt trước của KBTN).
Các tên hằng số của bản v2 được giữ để script và runner Kaggle của đội code không hỏng.
"""

from __future__ import annotations

# --- Bộ dữ liệu ------------------------------------------------------------------------------------
# tên trong code -> vai trò (giữ dạng dict tên -> chuỗi để `list(C.DATASETS)` và `.index()` vẫn chạy)
DATASETS = {
    "tcga_luad_lusc_primary_tumor": "large_n_reference",   # TCGA v2, RNA-seq, 515/501 sau QĐ22b; nguồn cho E7; E2–E3; E4
    "gse10245": "target_small",                             # GPL570, 37/18 sau QĐ22: cohort phân biệt phương pháp
    "gse30219": "target_main",                              # GPL570, 81/60 sau QĐ22 (gse30219_r5_fixed, R5)
    "gse37745": "target_main",                              # GPL570, 104/65 sau QĐ21 + QĐ22; E4–E5
    "gse42127": "target_second_platform",                   # GPL6884, 126/41 sau QĐ22; báo cáo riêng
}
# Thư mục chứa từng bộ (tương đối với $KLTN_DATA_ROOT). Bộ perm/ v2 giữ định dạng cũ; cohort GEO dùng
# experiment-data-geo-target/<thư mục> như RQ0′ (scripts/run_rq0_prime.py::cohort_data_dir).
PERM_DATA_DIR = "experiment-data-perm"
GEO_TARGET_DIR = "experiment-data-geo-target"
# v3.1: thư mục chính = layout Kaggle Input "experiment-data-v3" của đội code (DATA_AUDIT 27/10: gse30219 đã là bản R5,
# 146 mẫu). DATASET_DIR_ALTERNATIVES: tên thư mục thay thế được chấp nhận (layout RQ0′ trên máy cá nhân);
# survey_v2.data.resolve_dataset_dir() chọn thư mục đầu tiên tồn tại.
DATASET_DIRS = {
    "tcga_luad_lusc_primary_tumor": f"{PERM_DATA_DIR}/tcga_luad_lusc_primary_tumor",
    "gse10245": f"{GEO_TARGET_DIR}/gse10245",
    "gse30219": f"{GEO_TARGET_DIR}/gse30219",
    "gse37745": f"{GEO_TARGET_DIR}/gse37745",
    "gse42127": f"{GEO_TARGET_DIR}/gse42127",
}
DATASET_DIR_ALTERNATIVES = {"gse30219": (f"{GEO_TARGET_DIR}/gse30219_r5_fixed",)}
# Biến thể chỉ dùng cho độ nhạy (KHÔNG thuộc DATASETS, nên không vào vòng lặp nào của survey hay kaggle_jobs mặc định).
# gse30219_bas (QĐ53b, Review 08/11 Mục 8.5): GSE30219 R5 + 39 mẫu histology BAS gộp vào LUSC; scripts/build_gse30219_bas.py.
VARIANT_DATASET_DIRS = {"gse30219_bas": f"{GEO_TARGET_DIR}/gse30219_bas"}
# Số mẫu kỳ vọng (LUAD/LUSC) sau QC mẫu QĐ21 + QĐ22 + QĐ22b (29/10/2026). load_dataset() ghi vào notes và DỪNG khi
# KLTN_STRICT_N=1 (runner Kaggle đặt biến này) nếu lệch. Mẫu bị cách ly: <bộ>/excluded_samples_<bộ>.csv; n trước -> sau:
#   TCGA 1018 (517/501) -> 1016: 2 mẫu ghi nữ, gen Y mức nam (opposite, bỏ phiếu từng gen Y; QĐ22b)
#   GSE10245 58 (40/18) -> 55: 3 inconsistent (XIST và gen Y cùng cao; QĐ22)
#   GSE30219 146 (85/61) -> 141: 1 opposite + 4 inconsistent (QĐ22)
#   GSE37745 172 (106/66) -> 169: cặp GSM1019246/GSM1019247 (QĐ21) + 1 inconsistent (QĐ22)
#   GSE42127 176 (133/43) -> 167: 7 opposite + 2 inconsistent (QĐ22)
DATASET_EXPECTED_N = {
    "tcga_luad_lusc_primary_tumor": (515, 501),
    "gse10245": (37, 18),
    "gse30219": (81, 60),
    "gse37745": (104, 65),
    "gse42127": (126, 41),
    "gse30219_bas": (81, 99),       # biến thể: 185 (85/61 + 39 BAS) -> 180; QC trên cả biến thể chỉ loại lại đúng 5 mẫu R5
}
DATASET_PLATFORM = {"tcga_luad_lusc_primary_tumor": "RNA-seq (log2 TPM+1)", "gse10245": "GPL570", "gse30219": "GPL570",
                    "gse37745": "GPL570", "gse42127": "GPL6884", "gse30219_bas": "GPL570"}
# Tên giữ cho runner/test của đội code (commit 3059ce5, 2508a2a): cùng nội dung với DATASETS/DATASET_DIRS.
V3_EXPERIMENT_DATASETS = tuple(DATASETS)
V3_DATASET_DIRS = {name: tuple(path.split("/")) for name, path in DATASET_DIRS.items()}
SOURCE_DATASET = "tcga_luad_lusc_primary_tumor"            # nguồn (E7) và không gian gen G = giao với bộ này
GENE_SPACE_REFERENCE = SOURCE_DATASET                       # cohort GEO chỉ dùng gen có trong TCGA (B8)
GPL570_COHORTS = ("gse10245", "gse30219", "gse37745")       # k* gộp (QĐ8, QĐ11) chỉ trên ba cohort này
POOLED_KSTAR_COHORTS = GPL570_COHORTS
FEWSHOT_COHORTS = ("gse10245", "gse30219", "gse37745", "gse42127")   # E1b, E7
WRAPPER_DATASETS = GPL570_COHORTS                           # GA, BPSO chỉ chạy ở đây (3×5, WRAPPER_K)
RANKING_DATASET = "gse37745"                                # bộ dùng cho E5b (seed) và ma trận ROPE đầy đủ
TIMING_DATASET = SOURCE_DATASET                             # E2–E3: n lớn nhất, p = 19.938
STABILITY_DATASETS = (SOURCE_DATASET, "gse37745")           # E4
REDUNDANCY_DATASET = "gse37745"                             # E5a
RESULT_DIR = ("results", "survey_v2")                       # giữ tên để runner Kaggle của đội code không đổi
SPLITS_DIR = "splits"                                       # $KLTN_DATA_ROOT/splits/<bộ>/{repeated_splits.csv, fewshot_splits.csv}


def dataset_dir(name: str) -> str:
    """Đường dẫn tương đối (so với KLTN_DATA_ROOT) của một bộ; bộ không có trong registry -> perm/<name>."""
    return DATASET_DIRS.get(name) or VARIANT_DATASET_DIRS.get(name) or f"{PERM_DATA_DIR}/{name}"


def dataset_dir_candidates(name: str) -> tuple[str, ...]:
    """Thư mục chính rồi đến các thư mục thay thế (tương đối với KLTN_DATA_ROOT)."""
    return (dataset_dir(name),) + tuple(DATASET_DIR_ALTERNATIVES.get(name, ()))


# --- Không gian seed theo cohort ----------------------------------------------------------------------
# 100 + thứ tự trong DATASETS: giữ nguyên quy ước của đội code (commit 2508a2a, tests/test_v3_data_layout.py)
# để panel pha A trên TCGA chạy trước 29/10 có thể đối chiếu bằng kiểm tra bằng nhau (QĐ19). Bộ ngoài registry
# (bộ sanity, dữ liệu tổng hợp) dùng 99.
DATASET_SEED_IDS = {name: 100 + i for i, name in enumerate(DATASETS)}
DATASET_SEED_IDS["gse30219_bas"] = 200          # biến thể độ nhạy: mã riêng, không dùng lại seed của gse30219 (bản chia cũng mới)
UNREGISTERED_COHORT_ID = 99


def dataset_seed_id(name: str) -> int:
    """Mã cohort ổn định dùng trong seed_for(); lỗi nếu bộ không có trong registry."""
    try:
        return DATASET_SEED_IDS[name]
    except KeyError as exc:
        raise ValueError(f"unknown dataset for deterministic seed: {name}") from exc


def cohort_id(name: str) -> int:
    """Như dataset_seed_id() nhưng không lỗi với bộ ngoài registry (trả về UNREGISTERED_COHORT_ID)."""
    return DATASET_SEED_IDS.get(name, UNREGISTERED_COHORT_ID)


# --- Giao thức E1 (= RQ1) ------------------------------------------------------------------------------
REPEAT_SEEDS = (42, 43, 44, 45, 46)             # CV 5×5; trùng RQ0′; mọi phương pháp dùng chung lần chia
N_FOLDS = 5
WRAPPER_REPEATS = 3                             # wrapper chỉ 3 lần lặp đầu (3×5)
K_GRID = (1, 2, 3, 4, 5, 10, 25, 50)            # ngân sách k (QĐ2 + M4); vùng chính k <= 10
WRAPPER_K = (5, 10, 50)                         # GA, BPSO chỉ ở các k này
PREFILTER_P0 = 5000                             # lọc sơ bộ KHÔNG dùng nhãn: 5.000 gen phương sai lớn nhất trên train (GVHD chốt 26/10, KBTN M14)
N_RANDOM_DRAWS = 100                            # random-k: 100 lần rút mỗi fold
INNER_CV_FOLDS = 3                              # CV nội bộ để chọn C và ngưỡng
C_GRID = (0.01, 0.1, 1.0, 10.0)                 # lưới C của LR-L2 và SVM tuyến tính
SECONDARY_K = (5, 25)                           # bộ phân loại phụ chỉ ở k ∈ {5, 25}
RF_N_TREES = 500
CLASSIFIERS_MAIN = ("lr_l2",)                   # QĐ13: LR-L2 là bộ phân loại chính
CLASSIFIERS_SECONDARY = ("rf", "svm_linear")    # chỉ ở SECONDARY_K; bộ 11 mô hình mở rộng chỉ chạy với --models-extended
SMALL_N_FIXED_C = 1.0                           # M5: khi n_T <= SMALL_N_T_MAX mỗi lớp, C = 1 cố định, ngưỡng 0,5
SMALL_N_T_MAX = 10

# --- Phương pháp chọn gen (13) + tham chiếu -----------------------------------------------------------
FILTER_METHODS = ("f_test", "mutual_info", "mrmr", "relieff")
EMBEDDED_METHODS = ("l1_lr", "elastic_net", "rf_importance", "svm_rfe")
CONSTRAINED_METHODS = ("abess", "stability_selection", "ktsp")   # KBTN 5.1: L0, tần suất chọn, thứ hạng cặp
WRAPPER_METHODS = ("ga", "bpso")
NON_WRAPPER_METHODS = FILTER_METHODS + EMBEDDED_METHODS + CONSTRAINED_METHODS
ALL_SELECTORS = NON_WRAPPER_METHODS + WRAPPER_METHODS
REFERENCE_METHODS = ("b0", "random_k")
FIXED_PANEL_METHODS = ("ihc_2", "ihc_4")        # + "panel_<tên>" đọc từ survey_v2/panels/published_panels.csv
RANKING_METHODS = ("f_test", "mutual_info", "mrmr", "relieff", "rf_importance", "svm_rfe", "stability_selection")
PER_K_METHODS = ("l1_lr", "elastic_net", "abess", "ktsp", "ga", "bpso")
METHOD_FAMILY = {**{m: "filter" for m in FILTER_METHODS}, **{m: "embedded" for m in EMBEDDED_METHODS},
                 "abess": "constrained", "stability_selection": "constrained", "ktsp": "rank-based",
                 **{m: "wrapper" for m in WRAPPER_METHODS}, "b0": "reference", "random_k": "reference",
                 "ihc_2": "knowledge", "ihc_4": "knowledge"}
STANDARDIZE_BEFORE = ("l1_lr", "elastic_net", "svm_rfe", "abess", "stability_selection")   # z-score trong fold (M7)
METHOD_ID = {m: i for i, m in enumerate(ALL_SELECTORS)}   # dùng trong seed

# --- mRMR (Peng et al., 2005) ----------------------------------------------------------------------
MRMR_CRITERION = "MID"
MRMR_DISCRETIZE = "mean_half_sd"

# --- ReliefF ------------------------------------------------------------------------------------
RELIEFF_NEIGHBORS = 10

# --- Stability selection (Meinshausen & Bühlmann 2010) ------------------------------------------------
STABSEL_N_SUBSAMPLES = 100
STABSEL_FRACTION = 0.5
STABSEL_TARGET_NNZ_FACTOR = 2        # C chọn trên toàn bộ train sao cho ~ 2·k_max hệ số khác 0 (q của M&B)

# --- k-TSP (Tan et al., 2005) ----------------------------------------------------------------------
# Không tính điểm cặp khi p > KTSP_MAX_P (O(n p^2)); E3 bỏ điểm p lớn hơn. QĐ24 (30/10, v3.1.5): = P0 + số bản sao của E5a
# (5.000 + 50 = 5.050) vì E5a thêm 50 bản sao tương quan sau lọc sơ bộ; trước đó = 5.000 làm k-TSP lỗi ở E5a. E1, E1b, E3
# không đổi (p sau lọc = 5.000 ≤ cap; điểm p = 10.000 của E3 vẫn bị bỏ).
KTSP_MAX_P = 5000 + 50

# --- Panel tri thức (QĐ2, QĐ12; Yatabe et al., 2019) ---------------------------------------------------
IHC_2 = ("NKX2-1", "TP63")
IHC_4 = ("NKX2-1", "TP63", "NAPSA", "KRT5")
PUBLISHED_PANELS_CSV = "panels/published_panels.csv"   # tương đối với survey_v2/; sinh viên điền Girard 2016, Wilkerson 2013

# --- Wrapper ----------------------------------------------------------------------------------------
WRAPPER_POP = 30
WRAPPER_GENERATIONS = 50
WRAPPER_FITNESS_CV = 3
WRAPPER_FITNESS_C = 1.0
GA_TOURNAMENT = 3
GA_ELITE = 2
GA_MUTATION_PER_GENE = None
BPSO_W_START, BPSO_W_END = 0.9, 0.4
BPSO_C1, BPSO_C2 = 2.0, 2.0
BPSO_V_MAX = 6.0

# --- Few-shot (E1b, E7; giao thức RQ2 của KBTN v2) -------------------------------------------------------
FEWSHOT_SPLITS = 5                              # 5 lần chia test_T / pool_T, seed 42–46
FEWSHOT_TEST_FRACTION = 1.0 / 3.0               # test_T ≈ 1/3, phân tầng
FEWSHOT_SUBSETS = 4                             # 4 tập con mỗi mức n_T, lồng nhau theo n_T trong cùng (split, subset)
N_T_GRID = (5, 10, 20, 40, None)                # mẫu mỗi lớp; None = toàn bộ pool_T
FEWSHOT_K_GRID = (2, 3, 4, 5, 10)
FEWSHOT_N_T_LIMIT = {"gse10245": (5, 10)}       # KBTN: GSE10245 chỉ n_T ∈ {5, 10}
FEWSHOT_METHODS = NON_WRAPPER_METHODS           # 11 phương pháp (wrapper không chạy ở few-shot)
BOOTSTRAP_N = 2000                              # bootstrap hai tầng (bệnh nhân trong test_T × lần chạy)

# --- E7: chuyển giao đơn giản ---------------------------------------------------------------------------
TRANSFER_STRATEGIES = ("source_only", "select_on_source", "pooled", "target_only")
TRANSFER_N_T = (5, 10, 20)
TRANSFER_K_GRID = FEWSHOT_K_GRID
TRANSFER_SELECTOR = "l1_lr"                     # phương pháp chọn gen dùng cho cả ba chiến lược (và target-only ghép cặp)
PANEL_NORMALISATION = "within_sample_zscore"    # φ: z-score trong mẫu trên gen của panel; k = 2: hiệu hai gen

# --- E0: kiểm tra tín hiệu -------------------------------------------------------------------------------
E0_K = 10
E0_N_PERM = 200
E0_N_RANDOM = 100
E0_SELECTOR = "f_test"

# --- Độ ổn định (E4, E5) --------------------------------------------------------------------------------
STABILITY_M = 30
SUBSAMPLE_FRACTIONS = (0.5, 0.7, 0.9)
REDUNDANCY_RHO = (0.5, 0.8, 0.95)
REDUNDANCY_N_COPIES = 50
assert KTSP_MAX_P >= PREFILTER_P0 + REDUNDANCY_N_COPIES, "KTSP_MAX_P phải chứa P0 + số bản sao của E5a (QĐ24)"
REDUNDANCY_FRACTION = 0.7
SEED_VARIABILITY_SEEDS = (0, 1, 2, 3, 4)
SEED_VARIABILITY_METHODS = ("rf_importance", "stability_selection")   # + WRAPPER_METHODS ở WRAPPER_K
OVERLAP_K = 10                                  # E5c, ma trận ROPE, hình E4/E6 tại k = 10 (vùng chính)

# --- Thống kê ------------------------------------------------------------------------------------------------
ALPHA = 0.05
ROPE_AUROC = 0.01
NB_TEST_TRAIN_RATIO = 1.0 / 4.0
KSTAR_DELTA = 0.02                              # biên không thua kém (QĐ8)
KSTAR_SEQUENCE = (50, 25, 10, 5, 4, 3, 2, 1)    # chuỗi kiểm định cố định (QĐ11)
ROPE_K = (2, 5, 10)                             # cặp dẫn đầu được so bằng Bayesian ROPE ở các k này

# --- E2–E3: chi phí ---------------------------------------------------------------------------------------------
TIMING_REPEATS = 3
# QĐ18 (Review 29/10): E2/E3 được phép đo trên Kaggle với hiệu chuẩn giữa phiên: mỗi phiên đo lại 3 phương pháp mốc
# ở điểm E2 (k = TIMING_K[0]); phiên chỉ được chấp nhận khi wall time của mỗi mốc lệch < CALIBRATION_MAX_DRIFT so với phiên tham chiếu.
CALIBRATION_METHODS = ("l1_lr", "svm_rfe", "rf_importance")   # ≈ 10 s, 4 s, 32 s trên TCGA (n = 814, p = 5.000): đủ dài để độ lệch < 10 % có nghĩa
CALIBRATION_MAX_DRIFT = 0.10
TIMING_K = (10, 50)                             # hai điểm k (bản v2 là một điểm k = 50)
E3_N_GRID = (100, 200, 400, None)               # None = toàn bộ phần train (TCGA ≈ 814)
E3_P_GRID = (1000, 2000, 5000, 10000, None)     # None = toàn bộ gen (19.938)
# QĐ23 (29/10, v3.1.4): điểm p lớn nhất của E3 cho phương pháp quá chậm để một phiên Kaggle 11 giờ chứa đủ 3 lần đo
# (Elastic Net saga: ≈ 6.000 s mỗi lần ở p = 19.938 theo số mũ 1,39 đo ngày 28/10). Số mũ theo p fit trên các điểm còn lại;
# bài ghi rõ "đo tới p = 10.000". Giống cơ chế KTSP_MAX_P.
E3_MAX_P = {"elastic_net": 10000}
E3_WRAPPER_BUDGETS = ((10, 20), (30, 50))

# --- Sanity (Y10) --------------------------------------------------------------------------------------------------
SANITY_DATASET = "tcga_normal_vs_tumor_sanity"
SANITY_K = 10
SANITY_AUROC_THRESHOLD = 0.99


def seed_for(*parts: int) -> int:
    """Suy seed tất định từ (cohort_id, split_seed, fold, k, draw, ...) — sửa Y14."""
    import numpy as np
    return int(np.random.SeedSequence(list(int(p) for p in parts)).generate_state(1)[0] % (2**31 - 1))
