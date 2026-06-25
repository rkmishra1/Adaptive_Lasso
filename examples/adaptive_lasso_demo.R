# Adaptive Lasso Demo on Diabetes Dataset
# Demonstrates fitting the Adaptive Lasso and tuning hyper-parameters via CV.

library(lars)
library(glmnet)
source("R/adaptive_lasso.R")

# Create output folder
if (!dir.exists("outputs")) {
  dir.create("outputs")
}

# Load classic diabetes data
data(diabetes)
X <- as.matrix(diabetes$x)
y <- as.numeric(diabetes$y)

cat("==================================================\n")
cat("Adaptive Lasso Demo on Diabetes Data\n")
cat("==================================================\n\n")

# 1. Standard cross-validation for lambda (fixed gamma = 1)
cat("1. Running 5-fold CV to select optimal lambda (fixed gamma = 1.0)...\n")
set.seed(42)
cv_lamb <- cv.lambda(X, y, gamma = 1.0, k = 5)
cat(sprintf("Optimal lambda: %.4f\n\n", cv_lamb$lambda_min))

# 2. Joint cross-validation for lambda and gamma
cat("2. Running 5-fold joint CV for lambda and gamma...\n")
gamma_grid <- c(0.5, 1.0, 1.5, 2.0)
cv_joint <- cv.gamma(X, y, gamma_seq = gamma_grid, k = 5)

cat("\nJoint CV Results:\n")
cat(sprintf("Optimal Gamma: %.1f\n", cv_joint$gamma_min))
cat(sprintf("Optimal Lambda: %.4f\n", cv_joint$lambda_min))

# 3. Fit optimal model
cat("\n3. Fitting final model with optimal parameters...\n")
final_model <- fitadaplasso(X, y, gamma = cv_joint$gamma_min, tuning_seq = cv_joint$lambda_min)

# Print coefficients
cat("\nCoefficients at optimal parameters:\n")
coefs <- final_model$beta_lamb[, 1]
names(coefs) <- colnames(X)
print(round(coefs, 4))
cat(sprintf("Intercept: %.4f\n\n", final_model$intercept_vec[1]))

# 4. Save path plot
cat("4. Saving coefficient path plot across lambda sequence...\n")
model_path <- fitadaplasso(X, y, gamma = cv_joint$gamma_min, len_tuning = 50)
l1_norm <- colSums(abs(model_path$beta_lamb))

png("outputs/adaptive_lasso_path.png", width = 800, height = 600, res = 120)
matplot(l1_norm, t(model_path$beta_lamb), type = "o", lty = 1, pch = 20, col = 1:ncol(X),
        xlab = "L1 norm of coefficients", ylab = "Coefficients",
        main = sprintf("Adaptive Lasso Path (Gamma = %.1f)", cv_joint$gamma_min))
grid(lty = "dotted")
abline(h = 0, lty = "dashed", col = "gray50")
for (j in 1:ncol(X)) {
  text(max(l1_norm), model_path$beta_lamb[j, ncol(model_path$beta_lamb)], colnames(X)[j],
       pos = 4, col = j, cex = 0.8, xpd = TRUE)
}
dev.off()
cat("Saved plot to 'outputs/adaptive_lasso_path.png'\n")
cat("==================================================\n")
