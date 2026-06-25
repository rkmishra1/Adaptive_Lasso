# Adaptive Lasso Simulation Study: Zou (2006) Section 5 Model 1
# Compares standard Lasso and Adaptive Lasso on Model 1 (Sparse case) in terms of variable selection.

source("R/adaptive_lasso.R")
library(glmnet)

cat("=========================================================\n")
cat("Starting Adaptive Lasso Simulation: Zou (2006)\n")
cat("=========================================================\n\n")

# 1. Setup Simulation Parameters
set.seed(42)
n <- 100        # Sample size
p <- 8          # Number of predictors
rho <- 0.5      # Correlation parameter
sigma <- 3.0    # Noise standard deviation
n_reps <- 50    # Number of repetitions for quick demo (Zou used 200)

# True coefficient vector
beta_true <- c(3, 1.5, 0, 0, 2, 0, 0, 0)
active_indices <- c(1, 2, 5)
noise_indices <- c(3, 4, 6, 7, 8)

# Construct autoregressive covariance matrix Sigma
Sigma <- matrix(0, nrow = p, ncol = p)
for (i in 1:p) {
  for (j in 1:p) {
    Sigma[i, j] <- rho^abs(i - j)
  }
}

# Cholesky decomposition for generating correlated variables
L <- chol(Sigma)

# Trackers for results
correct_selected_adaptive <- 0
correct_selected_lasso <- 0

# Average number of zero coefficients correctly/incorrectly set to zero
c_adaptive_vec <- c()
i_adaptive_vec <- c()
c_lasso_vec <- c()
i_lasso_vec <- c()

cat(sprintf("Running %d simulation replications...\n", n_reps))

for (k in 1:n_reps) {
  # Generate predictors X ~ N(0, Sigma)
  Z <- matrix(rnorm(n * p), nrow = n, ncol = p)
  X <- Z %*% L
  colnames(X) <- paste0("X", 1:p)
  
  # Generate response
  y <- as.vector(X %*% beta_true + rnorm(n, mean = 0, sd = sigma))
  
  # A. Standard Lasso CV fit
  cv_lasso <- glmnet::cv.glmnet(X, y, alpha = 1)
  lasso_fit <- glmnet::glmnet(X, y, alpha = 1, lambda = cv_lasso$lambda.min)
  coef_lasso <- as.vector(coef(lasso_fit))[-1] # Exclude intercept
  
  # B. Adaptive Lasso CV fit
  cv_adap <- cv.gamma(X, y, gamma_seq = c(0.5, 1.0, 2.0), k = 5)
  fit_adap <- fitadaplasso(X, y, gamma = cv_adap$gamma_min, tuning_seq = cv_adap$lambda_min)
  coef_adap <- fit_adap$beta_lamb[, 1]
  
  # Variable selection metrics
  # C: number of zero coefficients correctly set to zero (max is 5)
  # I: number of non-zero coefficients incorrectly set to zero (max is 3)
  c_lasso <- sum(coef_lasso[noise_indices] == 0)
  i_lasso <- sum(coef_lasso[active_indices] == 0)
  c_lasso_vec <- c(c_lasso_vec, c_lasso)
  i_lasso_vec <- c(i_lasso_vec, i_lasso)
  
  c_adap <- sum(coef_adap[noise_indices] == 0)
  i_adap <- sum(coef_adap[active_indices] == 0)
  c_adaptive_vec <- c(c_adaptive_vec, c_adap)
  i_adaptive_vec <- c(i_adaptive_vec, i_adap)
  
  # Check if model selected is exactly correct
  if (c_lasso == 5 && i_lasso == 0) correct_selected_lasso <- correct_selected_lasso + 1
  if (c_adap == 5 && i_adap == 0) correct_selected_adaptive <- correct_selected_adaptive + 1
}

# Print results table
cat("\nSimulation Results (Zou JASA 2006 Table 1 Format):\n")
cat("-----------------------------------------------------------------\n")
cat(sprintf("%-20s | %-12s | %-12s | %-12s\n", "Method", "Mean Zero (C)", "Mean Nonzero (I)", "Prop. Correct"))
cat("-----------------------------------------------------------------\n")
cat(sprintf("%-20s | %-12.2f | %-12.2f | %-12.2f%%\n", 
            "Standard Lasso", mean(c_lasso_vec), mean(i_lasso_vec), (correct_selected_lasso / n_reps) * 100))
cat(sprintf("%-20s | %-12.2f | %-12.2f | %-12.2f%%\n", 
            "Adaptive Lasso", mean(c_adaptive_vec), mean(i_adaptive_vec), (correct_selected_adaptive / n_reps) * 100))
cat("-----------------------------------------------------------------\n")
cat("Note: C = zero coefficients correctly set to zero (max 5).\n")
cat("      I = active coefficients incorrectly set to zero (max 3).\n\n")
cat("Simulation study completed successfully!\n")
cat("=========================================================\n")
