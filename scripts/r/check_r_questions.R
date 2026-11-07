# Câu hỏi 1, 2 ở Mục 6 Review 07/11: kiểm hai gói R trên dữ liệu mô phỏng, chưa dùng dữ liệu đề tài.
#
# Câu 1: ptLasso có nhận mô hình giai đoạn 1 fit sẵn trên nguồn ngoài (Mục 2.4 của Craig et al.) không? Nếu không, bản cài bằng glmnet
# (offset và penalty.factor) phải trùng ptLasso trên ví dụ của chính gói, sai khác < 1e-6.
# Câu 2: thời gian một lần glmtrans với n_S = 1.016, p = 5.000, n_T = 5 và 20 mỗi lớp.
#
#   R_BASELINE_OUT=<thư mục kết quả> R_BASELINE_LIB=<thư mục cài gói> Rscript scripts/r/check_r_questions.R
#
# Đầu ra: r_questions.csv (mỗi bước: ok, giây, chi tiết hoặc lỗi cụ thể), ptlasso_source.txt, sessionInfo.txt.
# Mã thoát 0 khi mọi bước đạt, 1 khi có bước lỗi; các tệp đầu ra luôn được ghi trước khi thoát. Lỗi của glmtrans ở n_T nhỏ cũng là câu trả lời.

OUT <- Sys.getenv("R_BASELINE_OUT", "r_questions_out")
LIB <- Sys.getenv("R_BASELINE_LIB", file.path(tempdir(), "r_baselines_lib"))
CRAN <- "https://cloud.r-project.org"
EQUALITY_TOLERANCE <- 1e-6

dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
dir.create(LIB, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(LIB, .libPaths()))
options(repos = c(CRAN = CRAN), warn = 1)

rows <- list()

run_step <- function(step, fn) {
  t0 <- Sys.time()
  res <- tryCatch(list(ok = TRUE, detail = fn()), error = function(e) list(ok = FALSE, detail = conditionMessage(e)))
  seconds <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
  cat(sprintf("[%s] %s (%.1f giây): %s\n", if (res$ok) "OK" else "LOI", step, seconds, res$detail))
  rows[[length(rows) + 1]] <<- data.frame(step = step, ok = res$ok, seconds = seconds, detail = res$detail)
  invisible(res$ok)
}

install_one <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) utils::install.packages(pkg, lib = LIB, Ncpus = max(1L, parallel::detectCores()))
  if (!requireNamespace(pkg, quietly = TRUE)) stop("cài không thành công")
  paste("phiên bản", as.character(utils::packageVersion(pkg)))
}

# Dữ liệu của ví dụ binomial trong manual ptLasso 1.0.2: k = 3 nhóm, n = 300, p = 40.
example_data <- function() {
  k <- 3
  beta.common <- list(c(-.5, .5, .3, -.9, .1), c(-.3, .9, .1, -.1, .2), c(0.1, .2, -.1, .2, .3))
  beta.indiv <- lapply(1:k, function(i) 0.9 * beta.common[[i]])
  ptLasso::binomial.example.data(k = k, class.sizes = rep(100, k), scommon = 5, sindiv = rep(5, k), n = 300, p = 40,
                                 beta.common = beta.common, beta.indiv = beta.indiv)
}

# Câu 1a: đối số và mã nguồn của gói; mã nguồn ghi ra tệp để đọc lại không cần chạy lại.
q1_arguments <- function() {
  writeLines(deparse(ptLasso::ptLasso), file.path(OUT, "ptlasso_source.txt"))
  args <- names(formals(ptLasso::ptLasso))
  paste0("có đối số fitoverall: ", "fitoverall" %in% args, "; đối số: ", paste(args, collapse = ", "))
}

# Câu 1b: đưa cv.glmnet fit trên một tập ngoài vào fitoverall. Gói lấy offset từ fit.preval theo chỉ số mẫu của x, nên số mẫu khác là dấu hiệu không dùng được.
q1_external_model <- function() {
  set.seed(1)
  p <- 40
  x_source <- matrix(rnorm(400 * p), 400, p)
  y_source <- rbinom(400, 1, plogis(x_source[, 1] - x_source[, 2]))
  x_target <- matrix(rnorm(60 * p), 60, p)
  y_target <- rbinom(60, 1, plogis(x_target[, 1] - x_target[, 2]))
  groups <- rep(1:2, 30)
  external <- glmnet::cv.glmnet(cbind(x_source[, 1], x_source), y_source, family = "binomial", keep = TRUE)
  out <- tryCatch({
    fit <- ptLasso::ptLasso(x_target, y_target, groups = groups, alpha = 0.5, family = "binomial", type.measure = "deviance", fitoverall = external)
    paste0("gói không báo lỗi; số offset dùng từ fit.preval của nguồn = ", nrow(external$fit.preval), ", số mẫu đích = ", length(y_target),
           "; không nhận đúng nghĩa nếu hai số khác nhau (lớp ", class(fit)[1], ")")
  }, error = function(e) paste0("gói từ chối: ", conditionMessage(e)))
  out
}

# Câu 1c: lặp lại giai đoạn 2 bằng glmnet theo công thức của gói, từ chính fitoverall của gói, rồi so hệ số trên toàn đường lambda.
q1_glmnet_equals_package <- function() {
  set.seed(1234)
  data <- example_data()
  x <- data$x
  y <- data$y
  groups <- data$groups
  alpha <- 0.5
  fit <- ptLasso::ptLasso(x, y, groups = groups, alpha = alpha, family = "binomial", type.measure = "deviance", overall.lambda = "lambda.1se")
  overall <- fit$fitoverall
  lamhat <- overall$lambda.1se
  k <- length(unique(groups))
  preval <- overall$fit.preval[, which(overall$lambda == lamhat)]
  bhat <- as.numeric(coef(overall, s = lamhat, exact = FALSE))
  support <- which(bhat[-(1:k)] != 0)
  fac <- rep(1 / alpha, ncol(x))
  fac[support] <- 1
  # Không truyền lambda: cv.glmnet của gói fit toàn tập bằng đường lambda tự sinh; truyền sẵn đường đó là một phép tính khác,
  # vì glmnet ghi lambda đầu tiên bằng giá trị ngoại suy. Hai số còn lại chỉ để chẩn đoán khi không đạt.
  worst <- 0
  worst_given_lambda <- 0
  worst_selected <- 0
  for (g in seq_len(k)) {
    ix <- groups == g
    package_cv <- fit$fitpre[[g]]
    package_fit <- package_cv$glmnet.fit
    offset <- (1 - alpha) * preval[ix]
    own <- glmnet::glmnet(x[ix, ], y[ix], family = "binomial", offset = offset, penalty.factor = fac, standardize = TRUE)
    if (length(own$lambda) != length(package_fit$lambda)) {
      stop(sprintf("nhóm %d: đường lambda dài %d, của gói dài %d", g, length(own$lambda), length(package_fit$lambda)))
    }
    worst <- max(worst, max(abs(as.matrix(coef(own)) - as.matrix(coef(package_fit)))))
    given <- glmnet::glmnet(x[ix, ], y[ix], family = "binomial", offset = offset, penalty.factor = fac, lambda = package_fit$lambda, standardize = TRUE)
    worst_given_lambda <- max(worst_given_lambda, max(abs(as.matrix(coef(given)) - as.matrix(coef(package_fit)))))
    at <- package_cv$lambda.1se
    worst_selected <- max(worst_selected, max(abs(as.matrix(coef(given, s = at)) - as.matrix(coef(package_fit, s = at)))))
  }
  detail <- sprintf(paste0("sai khác lớn nhất trên đường lambda tự sinh: %.3e (ngưỡng %.0e); khi truyền sẵn lambda: %.3e, riêng tại lambda.1se: %.3e; ",
                           "%d gen support; fit.preval trong [%.3f, %.3f]"),
                    worst, EQUALITY_TOLERANCE, worst_given_lambda, worst_selected, length(support), min(preval), max(preval))
  if (!(worst < EQUALITY_TOLERANCE)) stop(detail)
  detail
}

# Câu 2: một lần glmtrans, phát hiện nguồn bật, tham số khác để mặc định của gói.
q2_glmtrans_time <- function(n_per_class) {
  force(n_per_class)
  function() {
    set.seed(n_per_class)
    p <- 5000
    n_source <- 1016
    signal <- c(rep(1, 10), rep(0, p - 10))
    make <- function(n) {
      x <- matrix(rnorm(n * p), n, p)
      y <- rep(0:1, length.out = n)
      x[y == 1, ] <- sweep(x[y == 1, , drop = FALSE], 2, signal, "+")
      list(x = x, y = y)
    }
    target <- make(2 * n_per_class)
    fit <- glmtrans::glmtrans(target = target, source = list(make(n_source)), family = "binomial", transfer.source.id = "auto")
    paste0("n_S = ", n_source, ", p = ", p, ", n_T = ", n_per_class, " mỗi lớp; nguồn được chọn: ", paste(fit$transfer.source.id, collapse = ","),
           "; số hệ số khác 0: ", sum(fit$beta[-1] != 0))
  }
}

all_ok <- TRUE
for (pkg in c("glmnet", "ptLasso", "glmtrans")) all_ok <- run_step(paste("cài", pkg), function() install_one(pkg)) && all_ok
all_ok <- run_step("câu 1a: đối số của ptLasso", q1_arguments) && all_ok
all_ok <- run_step("câu 1b: mô hình giai đoạn 1 fit sẵn trên nguồn ngoài", q1_external_model) && all_ok
all_ok <- run_step("câu 1c: glmnet trùng ptLasso trên ví dụ của gói", q1_glmnet_equals_package) && all_ok
for (n in c(5, 20)) all_ok <- run_step(paste0("câu 2: glmtrans n_T = ", n, " mỗi lớp"), q2_glmtrans_time(n)) && all_ok

utils::write.csv(do.call(rbind, rows), file.path(OUT, "r_questions.csv"), row.names = FALSE, fileEncoding = "UTF-8")
writeLines(c(paste("commit", Sys.getenv("REPO_COMMIT_HASH", "unknown")), paste("thời điểm", format(Sys.time(), tz = "UTC")),
             utils::capture.output(print(utils::sessionInfo()))), file.path(OUT, "sessionInfo.txt"))
quit(save = "no", status = if (all_ok) 0L else 1L)
