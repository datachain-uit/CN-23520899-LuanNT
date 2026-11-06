# PD-04 (Review 03/11, QĐ37): cài ptLasso và glmtrans, chạy ví dụ binomial trong manual CRAN, ghi sessionInfo().
# Chỉ kiểm tra môi trường trên dữ liệu mô phỏng của chính hai gói; chưa dùng dữ liệu của đề tài, chưa ép đúng k (M6).
#
#   R_BASELINE_OUT=<thư mục kết quả> R_BASELINE_LIB=<thư mục cài gói> Rscript scripts/r/check_r_baselines.R
#
# Đầu ra: r_baseline_check.csv (mỗi bước: ok, giây, chi tiết hoặc lỗi cụ thể), sessionInfo.txt, examples_output.txt.
# Mã thoát 0 khi mọi bước đạt, 1 khi có bước lỗi; các tệp đầu ra luôn được ghi trước khi thoát.

OUT <- Sys.getenv("R_BASELINE_OUT", "r_baselines_out")
LIB <- Sys.getenv("R_BASELINE_LIB", file.path(tempdir(), "r_baselines_lib"))
CRAN <- "https://cloud.r-project.org"
PACKAGES <- c("ptLasso", "glmtrans")

dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
dir.create(LIB, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(LIB, .libPaths()))
options(repos = c(CRAN = CRAN), warn = 1)

rows <- list()
example_log <- character()

run_step <- function(step, fn) {
  t0 <- Sys.time()
  res <- tryCatch(list(ok = TRUE, detail = fn()), error = function(e) list(ok = FALSE, detail = conditionMessage(e)))
  seconds <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
  cat(sprintf("[%s] %s (%.1f giây): %s\n", if (res$ok) "OK" else "LOI", step, seconds, res$detail))
  rows[[length(rows) + 1]] <<- data.frame(step = step, ok = res$ok, seconds = seconds, detail = res$detail)
  invisible(res$ok)
}

install_one <- function(pkg) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    return(paste("đã có sẵn, phiên bản", as.character(utils::packageVersion(pkg))))
  }
  warned <- character()
  withCallingHandlers(
    utils::install.packages(pkg, lib = LIB, Ncpus = max(1L, parallel::detectCores())),
    warning = function(w) {
      warned <<- c(warned, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste0("cài không thành công: ", paste(warned, collapse = " | ")))
  }
  paste("đã cài, phiên bản", as.character(utils::packageVersion(pkg)))
}

# Ví dụ binomial trong mục Examples của ptLasso (manual CRAN, bản 1.0.2): k = 3 nhóm, n = 300, p = 40.
example_ptlasso <- function() {
  suppressPackageStartupMessages(library(ptLasso))
  set.seed(1234)
  k <- 3
  class.sizes <- rep(100, k)
  scommon <- 5
  sindiv <- rep(5, k)
  n <- sum(class.sizes)
  p <- 2 * (sum(sindiv) + scommon)
  beta.common <- list(c(-.5, .5, .3, -.9, .1), c(-.3, .9, .1, -.1, .2), c(0.1, .2, -.1, .2, .3))
  beta.indiv <- lapply(1:k, function(i) 0.9 * beta.common[[i]])
  make <- function() {
    ptLasso::binomial.example.data(k = k, class.sizes = class.sizes, scommon = scommon, sindiv = sindiv, n = n, p = p,
                                   beta.common = beta.common, beta.indiv = beta.indiv)
  }
  out <- make()
  outtest <- make()
  fit <- ptLasso::ptLasso(out$x, out$y, groups = out$groups, alpha = 0.5, family = "binomial", type.measure = "auc")
  pred <- predict(fit, outtest$x, outtest$groups, ytest = outtest$y)
  example_log <<- c(example_log, "=== ptLasso, ví dụ binomial", utils::capture.output(print(pred)))
  metrics <- c(pretrained = pred$errpre[[1]], individual = pred$errind[[1]], overall = pred$erroverall[[1]])
  if (!all(is.finite(metrics))) stop("chỉ số không hữu hạn")
  paste0("AUC toàn bộ tập test: ", paste(names(metrics), round(metrics, 4), sep = " = ", collapse = "; "))
}

# Ví dụ logistic trong mục Examples của glmtrans (manual CRAN, bản 2.1.0): n.target = 100, K = 2 nguồn, p = 500.
example_glmtrans <- function() {
  suppressPackageStartupMessages(library(glmtrans))
  suppressPackageStartupMessages(library(glmnet))
  set.seed(0, kind = "L'Ecuyer-CMRG")
  train <- glmtrans::models("binomial", type = "all", n.target = 100, K = 2, p = 500)
  test <- glmtrans::models("binomial", type = "target", n.target = 500, p = 500)
  fit <- glmtrans::glmtrans(train$target, train$source, family = "binomial")
  y_transfer <- predict(fit, test$target$x, type = "class")
  fit_lasso <- glmnet::cv.glmnet(x = train$target$x, y = train$target$y, family = "binomial")
  y_lasso <- as.numeric(predict(fit_lasso, test$target$x, type = "class"))
  error_transfer <- mean(y_transfer != test$target$y)
  error_lasso <- mean(y_lasso != test$target$y)
  example_log <<- c(example_log, "=== glmtrans, ví dụ logistic", paste("nguồn được chọn:", paste(fit$transfer.source.id, collapse = ", ")),
                    paste("sai số glmtrans:", error_transfer), paste("sai số Lasso trên đích:", error_lasso))
  if (!all(is.finite(c(error_transfer, error_lasso)))) stop("chỉ số không hữu hạn")
  paste0("sai số phân loại: glmtrans = ", round(error_transfer, 4), "; Lasso trên đích = ", round(error_lasso, 4),
         "; nguồn được chọn: ", paste(fit$transfer.source.id, collapse = ", "))
}

installed <- vapply(PACKAGES, function(pkg) run_step(paste0("install_", pkg), function() install_one(pkg)), logical(1))
if (installed[["ptLasso"]]) run_step("example_ptLasso", example_ptlasso)
if (installed[["glmtrans"]]) run_step("example_glmtrans", example_glmtrans)

check <- do.call(rbind, rows)
utils::write.csv(check, file.path(OUT, "r_baseline_check.csv"), row.names = FALSE, fileEncoding = "UTF-8")
writeLines(example_log, file.path(OUT, "examples_output.txt"), useBytes = TRUE)
writeLines(c(paste("commit:", Sys.getenv("REPO_COMMIT_HASH", "không rõ")), paste("CRAN:", CRAN), paste("thời điểm:", format(Sys.time(), tz = "UTC", usetz = TRUE)), "",
             utils::capture.output(print(utils::sessionInfo()))),
           file.path(OUT, "sessionInfo.txt"), useBytes = TRUE)
print(check)
quit(save = "no", status = if (all(check$ok) && all(installed)) 0L else 1L)
