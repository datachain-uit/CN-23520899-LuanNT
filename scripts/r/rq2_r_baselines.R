# Baseline R của RQ2 (QĐ68, Review 08/11 Mục 8.3 Bảng 8 bước 3, 4, 7): ptLasso giai đoạn 1, 2 và Trans-GLM trên dữ liệu đề tài.
#
# Script CHỈ fit mô hình và ghi hệ số; mọi quy tắc chọn (α theo biên 0,01, β_eff, panel k gen, bổ sung F-test, cờ suy biến) và đánh giá
# nằm ở scripts/run_rq2_r_eval.py để test được bằng Python. Đầu vào do scripts/export_rq2_inputs.py ghi.
#
#   R_BASELINE_LIB=<thư mục cài gói> Rscript scripts/r/rq2_r_baselines.R <thư mục đầu vào> <thư mục kết quả> <stage1|all> [cohort ...]
#
# Giai đoạn 1 (mỗi không gian gen): cv.glmnet(Z_S, y_S, binomial, deviance, foldid cố định, standardize = FALSE); λ1 = lambda.1se.
# Giai đoạn 2 (mỗi lần chạy, α ∈ {0,75; 0,5; 0,25}): z-score L_T theo L_T (ddof = 0, sd = 0 -> 1); η1 = β10 + Z_T β1;
#   cv.glmnet(Z_T, y_T, binomial, offset = (1 − α) η1, penalty.factor = 1 trên S1 và 1/α ngoài S1, standardize = FALSE,
#   type.measure = "deviance", foldid trong L_T); λ2 = lambda.min. Không truyền sẵn đường λ (câu 1c).
# Trans-GLM: set.seed(r_seed); glmtrans(target = (Z_T, y_T), source = list((Z_S, y_S)), family = "binomial", transfer.source.id = "auto"),
#   tham số khác mặc định của gói.
# Dừng (mã 2) nếu packageVersion("ptLasso") != 1.0.2 hoặc packageVersion("glmtrans") != 2.1.0 (Y60). Lỗi của từng lần chạy được ghi lại,
# không dừng cả việc. Đầu ra: stage1_summary.csv, stage1_<space>.csv; <cohort>/{ptlasso_runs.csv, ptlasso_beta2.csv, transglm_runs.csv,
# transglm_beta.csv}; sessionInfo.txt.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("cần: <thư mục đầu vào> <thư mục kết quả> <stage1|all> [cohort ...]")
IN <- args[[1]]; OUT <- args[[2]]; STAGE <- args[[3]]
COHORTS <- if (length(args) > 3) args[-(1:3)] else list.dirs(IN, full.names = FALSE, recursive = FALSE)
COHORTS <- COHORTS[!startsWith(COHORTS, "space_") & file.exists(file.path(IN, COHORTS, "runs.csv"))]
if (!STAGE %in% c("stage1", "all")) stop("stage phải là stage1 hoặc all")
LIB <- Sys.getenv("R_BASELINE_LIB", file.path(tempdir(), "r_baselines_lib"))
CRAN <- "https://cloud.r-project.org"
PINNED <- c(ptLasso = "1.0.2", glmtrans = "2.1.0")
ALPHAS <- c(0.75, 0.5, 0.25)
N_CORES <- max(1L, parallel::detectCores())

dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
dir.create(LIB, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(LIB, .libPaths()))
options(repos = c(CRAN = CRAN), warn = 1)

install_pinned <- function(pkg, version) {
  have <- function() requireNamespace(pkg, quietly = TRUE) && as.character(utils::packageVersion(pkg)) == version
  if (!have()) utils::install.packages(pkg, lib = LIB, Ncpus = N_CORES)
  if (!have()) {               # CRAN đã có bản mới hơn: cài đúng bản từ kho lưu trữ của CRAN
    url <- sprintf("%s/src/contrib/Archive/%s/%s_%s.tar.gz", CRAN, pkg, pkg, version)
    utils::install.packages(url, lib = LIB, repos = NULL, type = "source")
  }
  if (!have()) {
    cat(sprintf("DỪNG: %s cần %s, có %s\n", pkg, version,
                if (requireNamespace(pkg, quietly = TRUE)) as.character(utils::packageVersion(pkg)) else "không cài được"))
    quit(save = "no", status = 2L)
  }
}
if (!requireNamespace("glmnet", quietly = TRUE)) utils::install.packages("glmnet", lib = LIB, Ncpus = N_CORES)
for (pkg in names(PINNED)) install_pinned(pkg, PINNED[[pkg]])
suppressPackageStartupMessages({ library(glmnet); library(glmtrans) })

read_matrix <- function(path, row_names = FALSE) {
  df <- utils::read.csv(path, check.names = FALSE, row.names = if (row_names) 1 else NULL)
  as.matrix(df)
}
bind_rows <- function(x) {                          # rbind bỏ qua NULL; không còn dòng nào -> bảng rỗng thay vì NULL
  d <- do.call(rbind, x)
  if (is.null(d)) data.frame() else d
}
zscore <- function(X) {
  mu <- colMeans(X)
  sd <- sqrt(colMeans(sweep(X, 2, mu)^2))          # ddof = 0 như zscore_fit của Python
  sd[!is.finite(sd) | sd <= 0] <- 1
  sweep(sweep(X, 2, mu), 2, sd, "/")
}

# ------------------------------------------------------------------------------------------------- giai đoạn 1
spaces <- unique(vapply(COHORTS, function(c) trimws(readLines(file.path(IN, c, "space.txt"), warn = FALSE)[1]), ""))
if (STAGE == "stage1") spaces <- sort(list.dirs(IN, full.names = FALSE, recursive = FALSE)[startsWith(list.dirs(IN, full.names = FALSE, recursive = FALSE), "space_")])
source_data <- list(); stage1 <- list(); summary_rows <- list()
for (sp in spaces) {
  t0 <- Sys.time()
  Z <- read_matrix(file.path(IN, sp, "source_z.csv.gz"))
  y <- utils::read.csv(file.path(IN, sp, "source_y.csv"))$y
  foldid <- utils::read.csv(file.path(IN, sp, "source_foldid.csv"))$foldid
  cv1 <- glmnet::cv.glmnet(Z, y, family = "binomial", type.measure = "deviance", foldid = foldid, standardize = FALSE)
  b <- as.matrix(stats::coef(cv1, s = "lambda.1se"))[, 1]
  beta1 <- b[-1]; names(beta1) <- colnames(Z)
  source_data[[sp]] <- list(Z = Z, y = y)
  stage1[[sp]] <- list(b0 = unname(b[1]), beta1 = beta1)
  utils::write.csv(data.frame(gene = c("(Intercept)", colnames(Z)), beta1 = unname(b)), file.path(OUT, paste0("stage1_", sp, ".csv")), row.names = FALSE)
  i1se <- which(cv1$lambda == cv1$lambda.1se)
  summary_rows[[sp]] <- data.frame(space = sp, n_source = nrow(Z), p = ncol(Z), lambda_1se = cv1$lambda.1se, lambda_min = cv1$lambda.min,
                                   cvm_1se = cv1$cvm[i1se], size_S1 = sum(beta1 != 0), seconds = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1))
  cat(sprintf("[stage1] %s: n = %d, p = %d, lambda.1se = %.4g, |S1| = %d\n", sp, nrow(Z), ncol(Z), cv1$lambda.1se, sum(beta1 != 0)))
}
utils::write.csv(do.call(rbind, summary_rows), file.path(OUT, "stage1_summary.csv"), row.names = FALSE)

# ------------------------------------------------------------------------------------------------- giai đoạn 2 và Trans-GLM
fit_run <- function(run, Xall, yall, members, s1, src) {
  m <- members[members$run_id == run$run_id, ]
  tr <- m[m$role == "train", ]
  Zt <- zscore(Xall[tr$sample_id, , drop = FALSE])
  yt <- yall[tr$sample_id]
  eta1 <- as.numeric(s1$b0 + Zt %*% s1$beta1)
  in_s1 <- s1$beta1 != 0
  pt_runs <- list(); pt_beta <- list()
  for (a in ALPHAS) {
    t0 <- Sys.time()
    res <- tryCatch({
      cv2 <- glmnet::cv.glmnet(Zt, yt, family = "binomial", offset = (1 - a) * eta1, penalty.factor = ifelse(in_s1, 1, 1 / a),
                               standardize = FALSE, type.measure = "deviance", foldid = tr$foldid)
      b2 <- as.matrix(stats::coef(cv2, s = "lambda.min"))[, 1]
      nz <- which(b2[-1] != 0)
      list(row = data.frame(run_id = run$run_id, alpha = a, lambda_min = cv2$lambda.min, cvm_min = cv2$cvm[which(cv2$lambda == cv2$lambda.min)],
                            beta2_0 = unname(b2[1]), nnz_beta2 = length(nz), error = NA_character_),
           beta = if (length(nz)) data.frame(run_id = run$run_id, alpha = a, gene = colnames(Zt)[nz], beta2 = unname(b2[-1][nz])) else NULL)
    }, error = function(e) list(row = data.frame(run_id = run$run_id, alpha = a, lambda_min = NA, cvm_min = NA, beta2_0 = NA, nnz_beta2 = NA,
                                                 error = conditionMessage(e)), beta = NULL))
    res$row$seconds <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 2)
    pt_runs[[length(pt_runs) + 1]] <- res$row
    pt_beta[[length(pt_beta) + 1]] <- res$beta
  }
  t0 <- Sys.time()
  tg <- tryCatch({
    set.seed(run$r_seed)
    fit <- glmtrans::glmtrans(target = list(x = Zt, y = yt), source = list(list(x = src$Z, y = src$y)), family = "binomial",
                              transfer.source.id = "auto")
    b <- as.numeric(fit$beta)
    nz <- which(b[-1] != 0)
    list(row = data.frame(run_id = run$run_id, transfer_source_id = paste(fit$transfer.source.id, collapse = ";"), nnz = length(nz),
                          intercept = b[1], error = NA_character_),
         beta = if (length(nz)) data.frame(run_id = run$run_id, gene = colnames(Zt)[nz], beta = b[-1][nz]) else NULL)
  }, error = function(e) list(row = data.frame(run_id = run$run_id, transfer_source_id = NA, nnz = NA, intercept = NA, error = conditionMessage(e)),
                              beta = NULL))
  tg$row$seconds <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 2)
  list(pt_runs = do.call(rbind, pt_runs), pt_beta = do.call(rbind, pt_beta), tg_row = tg$row, tg_beta = tg$beta)
}

if (STAGE == "all") {
  for (cohort in COHORTS) {
    t0 <- Sys.time()
    sp <- trimws(readLines(file.path(IN, cohort, "space.txt"), warn = FALSE)[1])
    Xall <- read_matrix(file.path(IN, cohort, "target_x.csv.gz"), row_names = TRUE)
    yy <- utils::read.csv(file.path(IN, cohort, "target_y.csv"), colClasses = c(sample_id = "character"))
    yall <- stats::setNames(yy$y, yy$sample_id)
    runs <- utils::read.csv(file.path(IN, cohort, "runs.csv"), colClasses = c(run_id = "character", n_t = "character"))
    members <- utils::read.csv(file.path(IN, cohort, "membership.csv"), colClasses = c(run_id = "character", sample_id = "character"))
    stopifnot(identical(colnames(Xall), names(stage1[[sp]]$beta1)))
    res <- parallel::mclapply(seq_len(nrow(runs)), function(i) fit_run(runs[i, ], Xall, yall, members, stage1[[sp]], source_data[[sp]]),
                              mc.cores = N_CORES, mc.preschedule = FALSE)
    bad <- vapply(res, inherits, TRUE, what = "try-error")
    if (any(bad)) stop(sprintf("%s: %d lần chạy lỗi ở mức tiến trình: %s", cohort, sum(bad), as.character(res[bad][[1]])))
    od <- file.path(OUT, cohort)
    dir.create(od, recursive = TRUE, showWarnings = FALSE)
    utils::write.csv(bind_rows(lapply(res, `[[`, "pt_runs")), file.path(od, "ptlasso_runs.csv"), row.names = FALSE)
    utils::write.csv(bind_rows(lapply(res, `[[`, "pt_beta")), file.path(od, "ptlasso_beta2.csv"), row.names = FALSE)
    utils::write.csv(bind_rows(lapply(res, `[[`, "tg_row")), file.path(od, "transglm_runs.csv"), row.names = FALSE)
    utils::write.csv(bind_rows(lapply(res, `[[`, "tg_beta")), file.path(od, "transglm_beta.csv"), row.names = FALSE)
    cat(sprintf("[stage2] %s: %d lần chạy, %.1f phút\n", cohort, nrow(runs), as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  }
}

writeLines(c(paste("commit", Sys.getenv("REPO_COMMIT_HASH", "unknown")), paste("stage", STAGE), paste("thời điểm", format(Sys.time(), tz = "UTC")),
             paste("cohort", paste(COHORTS, collapse = ",")), paste("số nhân", N_CORES),
             utils::capture.output(print(utils::sessionInfo()))), file.path(OUT, "sessionInfo.txt"))
cat("R_BASELINES_DONE\n")
