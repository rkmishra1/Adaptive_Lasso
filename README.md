# 📐 Adaptive Lasso in R

[![R Version](https://img.shields.io/badge/R-%3E%3D%204.0-blue.svg)](https://www.r-project.org/)
[![Build Status](https://img.shields.io/badge/Tests-Passing-success.svg)](file:///Users/ramakrushnamishra/Documents/antigravity/noble-franklin/Adaptive_Lasso/tests/test_adaptive_lasso.R)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An elegant, clean, and bug-free implementation of the **Adaptive Lasso** algorithm in R, complete with cross-validation routines for selecting optimal hyperparameters (both tuning parameter $\lambda$ and adaptive power parameter $\gamma$).

---

## 🚀 Features

- **Oracle Properties**: Unlike the standard Lasso, the Adaptive Lasso has the oracle properties, meaning it performs variable selection as if the true active set were known beforehand, leading to unbiased coefficient estimation.
- **Auto-weighting**: Computes adaptive weights from OLS (for $p < n$) or Ridge regression (for $p \ge n$) automatically.
- **Coordinate Descent Solver**: High-performance implementation of coordinate descent with soft-thresholding.
- **Hyperparameter CV Tuning**: Routines for K-fold cross-validation on $\lambda$ and joint cross-validation on both $\lambda$ and $\gamma$.
- **Validation Suite**: Unit tests to verify correct standardization, weights dimension, and solver convergence.

---

## 📁 Directory Structure

```
├── README.md                 # Project math explanation and user guide
├── R/
│   └── adaptive_lasso.R      # Core implementation (fitting, scaling, CV)
├── tests/
│   └── test_adaptive_lasso.R # Unit test suite verifying functions
└── examples/
    └── adaptive_lasso_demo.R # Demo script fitting Adaptive Lasso on Diabetes data
```

---

## 🧠 Theoretical Background & Math

The Adaptive Lasso, introduced by Hui Zou in 2006, modifies the standard Lasso by applying weights to each coefficient in the $L_1$ penalty:
$$\min_{\beta} \left\{ \frac{1}{2n} \sum_{i=1}^n \left( y_i - \beta_0 - \sum_{j=1}^p X_{ij} \beta_j \right)^2 + \lambda \sum_{j=1}^p w_j |\beta_j| \right\}$$

### Weight Computation
The weights $w_j$ are defined as:
$$w_j = \frac{1}{|\hat{\beta}_{\text{ini}}|^\gamma}$$
where $\hat{\beta}_{\text{ini}}$ is an initial consistent estimator:
- **OLS Estimator**: Used if $p < n$.
- **Ridge Estimator**: Used if $p \ge n$ (regularized inversion to handle collinearity).
The parameter $\gamma > 0$ controls the penalty skew (typically $\gamma = 1.0$ or $\gamma = 2.0$).

### Coordinate Descent Algorithm
To solve for $\beta$ for a given $\lambda$ and weights $w$:
1. Transform $X$ to the weighted space: $X^*_j = X_j / w_j$.
2. Perform standard Lasso on the weighted space using coordinate descent.
3. Transform the estimated coefficients back: $\hat{\beta}_j = \hat{\beta}^*_j / w_j$.

### Workflow Diagram

```mermaid
graph TD
    A[Start: Inputs X, y, gamma] --> B{Rank X < p?}
    B -- Yes --> C[Fit OLS to get beta_ini]
    B -- No --> D[Fit Ridge to get beta_ini]
    C --> E[Compute weights w = 1 / |beta_ini|^gamma]
    D --> E
    E --> F[Transform X_w = X / w]
    F --> G[Standardize & Center Ystd, Xstd]
    G --> H[Coordinate Descent with Soft-thresholding]
    H --> I[Obtain beta_std]
    I --> J[Scale back: beta = beta_std / w]
    J --> K[Recover original Intercept]
    K --> L[End]
```

---

## 🏁 Quick Start & Usage

Ensure you have the `glmnet` and `lars` packages installed in R.

### Running the Demo Script
To run the joint cross-validation and print the selected coefficients:
```bash
Rscript examples/adaptive_lasso_demo.R
```
This generates the coefficient path plot saved to:
- `outputs/adaptive_lasso_path.png`

### Core R Code Usage
```R
source("R/adaptive_lasso.R")

# Load your X and y
X <- as.matrix(your_predictors)
y <- as.numeric(your_response)

# Run 5-fold cross-validation to select best lambda and gamma
cv_res <- cv.gamma(X, y, gamma_seq = c(0.5, 1.0, 2.0), k = 5)
cat("Best Gamma:", cv_res$gamma_min)
cat("Best Lambda:", cv_res$lambda_min)

# Fit final model with optimal parameters
fit <- fitadaplasso(X, y, gamma = cv_res$gamma_min, tuning_seq = cv_res$lambda_min)
print(fit$beta_lamb) # Coefficients
print(fit$intercept_vec) # Intercepts
```

---

## ✅ Verification Results

You can run the verification tests to check the codebase:
```bash
Rscript tests/test_adaptive_lasso.R
```

<details>
<summary><b>Click to expand verification log</b></summary>

```
==========================================
Starting Adaptive Lasso Verification Tests
==========================================

Test 1: Soft-thresholding...
  Soft thresholding works: TRUE 

Test 2: scale_X weights dimensions...
  scale_X output dimensions correct: TRUE 

Test 3: standardize outputs mean and norm...
  Standardization centering correct: TRUE 

Test 4: Coordinate descent single lambda fit...
  Coordinate descent completes and converges: TRUE 

Test 5: Full sequence path fit...
  Sequence fitting completes and outputs correct dimensions: TRUE 

==========================================
SUCCESS: All Adaptive Lasso tests passed!
==========================================
```
</details>
