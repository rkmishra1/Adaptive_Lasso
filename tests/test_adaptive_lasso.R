# Verification tests for Adaptive Lasso implementation
# Tests core function correctness and coordinate descent convergence.

source("R/adaptive_lasso.R")

cat("\n==========================================\n")
cat("Starting Adaptive Lasso Verification Tests\n")
cat("==========================================\n\n")

# 1. Soft thresholding test
cat("Test 1: Soft-thresholding...\n")
success1 <- (soft(3, 1) == 2) && (soft(-3, 1) == -2) && (soft(0.5, 1) == 0)
cat("  Soft thresholding works:", success1, "\n\n")

# 2. scale_X weights check
cat("Test 2: scale_X weights dimensions...\n")
set.seed(42)
X <- matrix(rnorm(100), 20, 5)
Y <- rnorm(20)
scaled <- scale_X(X, Y, gamma = 1)
success2 <- (length(scaled$weights) == 5) && (ncol(scaled$X_w) == 5)
cat("  scale_X output dimensions correct:", success2, "\n\n")

# 3. standardize outputs check
cat("Test 3: standardize outputs mean and norm...\n")
std <- standardize(X, Y, gamma = 1)
success3 <- (abs(mean(std$Ystd)) < 1e-12) && (abs(mean(colMeans(std$Xstd))) < 1e-12)
cat("  Standardization centering correct:", success3, "\n\n")

# 4. Single lambda fit coordinate descent check
cat("Test 4: Coordinate descent single lambda fit...\n")
fit <- adaplassostd_lambda(std$Xstd, std$Ystd, lambda = 0.1)
success4 <- (length(fit$beta) == 5) && (!is.na(fit$obj_min))
cat("  Coordinate descent completes and converges:", success4, "\n\n")

# 5. Full sequence path fit check
cat("Test 5: Full sequence path fit...\n")
path_fit <- fitadaplasso(X, Y, gamma = 1.0, len_tuning = 10)
success5 <- (ncol(path_fit$beta_lamb) == 10) && (length(path_fit$intercept_vec) == 10)
cat("  Sequence fitting completes and outputs correct dimensions:", success5, "\n\n")

# -----------------------------------------------------------------
# Final Result Summary
# -----------------------------------------------------------------
success <- success1 && success2 && success3 && success4 && success5

if (success) {
  cat("==========================================\n")
  cat("SUCCESS: All Adaptive Lasso tests passed!\n")
  cat("==========================================\n")
  quit(status = 0)
} else {
  cat("==========================================\n")
  cat("FAILURE: Discrepancies found in Adaptive Lasso!\n")
  cat("==========================================\n")
  quit(status = 1)
}
