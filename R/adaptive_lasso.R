#' scale_X Scales a matrix of inputs according to LARS algorithm
#'
#' @param X n x p design matrix of inputs
#' @param Y n x 1 vector of outputs
#' @param gamma a scalar(>0) input used in the weight(user input)
#'
#' @return A list with the elements:
#' \item{X_w}{A n x p matrix scaled according to LARS algorithm}
#' \item{weights}{adaptive weights}
#' @export
scale_X <- function(X, Y, gamma) {
  # Computing the best lambda for the Ridge estimator
  cv_mod <- glmnet::cv.glmnet(X, Y, alpha = 0)
  # Optimal value of lambda for Ridge
  lambda_min <- cv_mod$lambda.min
  p <- ncol(X)
  n <- nrow(X)
  
  if (Matrix::rankMatrix(X) < p) {
    # Calculating ridge estimator
    beta <- solve(crossprod(X) + lambda_min * diag(p)) %*% crossprod(X, Y)
  } else {
    # Calculating OLS
    beta <- solve(t(X) %*% X) %*% t(X) %*% Y
  }
  
  weights <- as.vector(abs(beta)^gamma)
  # Handle zero weights to avoid division by zero
  weights[weights == 0] <- 1e-8
  
  # Scales X
  X_w <- t(t(X) / weights)
  return(list(X_w = X_w, weights = weights))
}

#' Standardizes the input design matrix X and output vector Y
#'
#' @param X n x p design matrix of inputs
#' @param Y n x 1 vector of outputs
#' @param gamma a scalar(>0) input used in the weight
#'
#' @return A list with:
#' \item{Xstd}{scaled X}
#' \item{Ystd}{scaled Y}
#' \item{meanY}{mean of Y}
#' \item{meanX}{Column means after centering the weighted X matrix}
#' \item{weights}{weights obtained by centering X_w}
#' @export
standardize <- function(X, Y, gamma) {
  n <- length(Y)
  p <- ncol(X)
  # Scaling X
  Xstd <- scale_X(X, Y, gamma)$X_w
  meanX <- colMeans(Xstd)
  Xcentered <- scale(Xstd, scale = FALSE)
  weights <- sqrt(diag(crossprod(Xcentered) / n))
  weights[weights == 0] <- 1
  Xstd <- t(t(Xcentered) / weights)
  # Center Y
  meanY <- mean(Y)
  Ystd <- Y - meanY
  
  return(list(Xstd = Xstd, Ystd = Ystd, meanY = meanY, meanX = meanX, weights = weights))
}

#' Soft-thresholding of a scalar a at level lambda
#'
#' @param a scalar to be soft-thresholded
#' @param lambda level of soft thresholding
#' @return soft-thresholded value
#' @export
soft <- function(a, lambda) {
  if (a > lambda) {
    return(a - lambda)
  } else if (a < (-lambda)) {
    return(a + lambda)
  } else {
    return(0)
  }
}

#' Compute the Objective Function for Adaptive Lasso
#'
#' @param Xstd n x p standardized design matrix X
#' @param Ystd n x 1 centered output vector
#' @param beta p x 1 vector of parameters
#' @param lambda tuning parameter
#' @return Objective function value
#' @export
adaplasso <- function(Xstd, Ystd, beta, lambda) {
  n <- length(Ystd)
  obj <- sum((Ystd - (Xstd %*% beta))^2) / (2 * n) + sum(lambda * abs(beta))
  return(obj)
}

#' Fits Adaptive Lasso on standardized data for a given lambda
#'
#' @param Xstd n x p standardized design matrix
#' @param Ystd n x 1 centered response vector
#' @param lambda tuning parameter
#' @param beta_init optional starting point for coordinate descent
#' @param eps precision level for convergence
#' @return A list containing estimated beta coefficients and minimum objective value.
#' @export
adaplassostd_lambda <- function(Xstd, Ystd, lambda, beta_init = NULL, eps = 0.001) {
  n <- length(Ystd)
  p <- ncol(Xstd)
  
  if (nrow(Xstd) != length(Ystd)) {
    stop("Error: nrow(X) and length(Y) are not equal")
  }
  if (lambda < 0) {
    stop("Error: Lambda is negative")
  }
  if (is.null(beta_init)) {
    beta_init <- rep(0, p)
  } else if (length(beta_init) != p) {
    stop("Error: dimension of beta_init and ncol(X) do not match")
  }
  
  beta <- beta_init
  curr_obj <- adaplasso(Xstd, Ystd, beta, lambda)
  last_obj <- Inf
  r <- Ystd - Xstd %*% beta_init
  
  while ((last_obj - curr_obj) > eps) {
    for (j in 1:p) {
      beta_old <- beta[j]
      beta[j] <- soft(beta[j] + (crossprod(Xstd[, j], r)) / n, lambda)
      r <- r + Xstd[, j] * (beta_old - beta[j])
    }
    last_obj <- curr_obj
    curr_obj <- adaplasso(Xstd, Ystd, beta, lambda)
  }
  
  return(list(beta = beta, obj_min = curr_obj))
}

#' Fits Adaptive Lasso on a sequence of lambda values based on standardized data
#'
#' @param Xstd n x p standardized design matrix
#' @param Ystd n x 1 centered response vector
#' @param tuning_seq optional sequence of tuning parameters
#' @param len_tuning length of desired sequence
#' @param eps convergence precision
#' @return A list containing tuning sequence, beta coefficients matrix, and minimum objective values.
#' @export
adaplassostdseq_lambda <- function(Xstd, Ystd, tuning_seq = NULL, len_tuning = 60, eps = 0.001) {
  n <- length(Ystd)
  if (nrow(Xstd) != length(Ystd)) {
    stop("Dimensions of X and Y do not match")
  }
  
  if (!is.null(tuning_seq)) {
    tuning_seq <- sort(tuning_seq[tuning_seq >= 0], decreasing = TRUE)
    if (length(tuning_seq) == 0) {
      warning("Sequence of tuning parameters for fixed gamma not supplied")
      tuning_seq <- NULL
    } else {
      len_tuning <- length(tuning_seq)
    }
  }
  
  if (is.null(tuning_seq)) {
    lambda_max <- max(abs(crossprod(Xstd, Ystd)) / n)
    tuning_seq <- exp(seq(log(lambda_max), log(0.005), length = len_tuning))
  }
  
  p <- ncol(Xstd)
  beta <- rep(0, p)
  beta_lamb <- matrix(0, p, len_tuning)
  obj_min_vec <- rep(0, len_tuning)
  
  for (i in 1:len_tuning) {
    fit <- adaplassostd_lambda(Xstd, Ystd, tuning_seq[i], beta_init = beta, eps = eps)
    beta_lamb[, i] <- fit$beta
    obj_min_vec[i] <- fit$obj_min
    beta <- fit$beta
  }
  
  return(list(tuning_seq = tuning_seq, beta_lamb = beta_lamb, obj_min_vec = obj_min_vec))
}

#' Fits Adaptive Lasso on original data
#'
#' @param X n x p design matrix of covariates
#' @param Y n x 1 response vector
#' @param tuning_seq optional sequence of tuning parameters
#' @param len_tuning length of desired sequence
#' @param gamma scaling power parameter
#' @param eps convergence precision
#' @return A list containing the tuning sequence, beta coefficients matrix, and unscaled intercepts.
#' @export
fitadaplasso <- function(X, Y, tuning_seq = NULL, len_tuning = 60, gamma = 0.01, eps = 0.001) {
  sc <- scale_X(X, Y, gamma)
  Std <- standardize(X, Y, gamma)
  
  fit <- adaplassostdseq_lambda(Std$Xstd, Std$Ystd, tuning_seq, len_tuning, eps)
  tuning_seq <- fit$tuning_seq
  
  beta <- fit$beta_lamb
  # Scale back coefficients
  beta_lamb <- beta / (sc$weights * Std$weights)
  # Recalculate intercept vector
  intercept_vec <- Std$meanY - as.vector(t(Std$meanX) %*% beta_lamb)
  
  return(list(tuning_seq = tuning_seq, beta_lamb = beta_lamb, intercept_vec = intercept_vec))
}

#' Perform k-fold cross-validation to select optimal lambda for a fixed gamma
#'
#' @param X design matrix
#' @param Y response vector
#' @param tuning_seq optional sequence of lambdas
#' @param len_tuning length of lambdas sequence
#' @param gamma scaling power parameter
#' @param k number of folds
#' @param id_fold optional folds assignment vector
#' @param eps convergence precision
#' @return A list of CV statistics, optimal lambda, and model details.
#' @export
cv.lambda <- function(X, Y, tuning_seq = NULL, len_tuning = 60, gamma = 0.01, k = 5, id_fold = NULL, eps = 0.001) {
  n <- length(Y)
  fit_adaplasso <- fitadaplasso(X, Y, tuning_seq = tuning_seq, len_tuning = len_tuning, gamma = gamma, eps = eps)
  
  if (is.null(id_fold)) {
    id_fold <- sample(1:n, size = n) %% k + 1
  }
  
  tuning_seq <- fit_adaplasso$tuning_seq
  len_tuning <- length(tuning_seq)
  cv_folds <- matrix(NA, k, len_tuning)
  
  for (fold in 1:k) {
    Xtrain <- X[id_fold != fold, ]
    Ytrain <- Y[id_fold != fold]
    
    Xtest <- X[id_fold == fold, ]
    Ytest <- Y[id_fold == fold]
    
    adaplasso <- fitadaplasso(Xtrain, Ytrain, tuning_seq = tuning_seq, len_tuning = len_tuning, gamma = gamma, eps = eps)
    
    # Calculate RSS on test fold
    y_pred <- t(c(adaplasso$intercept_vec) + t(Xtest %*% adaplasso$beta_lamb))
    cv_folds[fold, ] <- colSums((Ytest - y_pred)^2) / length(Ytest)
  }
  
  cv <- colMeans(cv_folds)
  min_idx <- which.min(cv)
  lambda_min <- tuning_seq[min_idx]
  
  return(list(
    tuning_seq = tuning_seq,
    beta_lamb = fit_adaplasso$beta_lamb,
    intercept_vec = fit_adaplasso$intercept_vec,
    id_fold = id_fold,
    lambda_min = lambda_min,
    cv = cv
  ))
}

#' Cross-Validation to choose the optimal gamma and lambda
#'
#' @param X design matrix
#' @param Y response vector
#' @param tuning_seq optional sequence of lambdas
#' @param len_tuning length of lambdas sequence
#' @param gamma_seq optional sequence of gammas
#' @param n_gamma length of gammas sequence
#' @param k number of folds
#' @param id_fold optional folds assignment vector
#' @param eps convergence precision
#' @return A list containing CV matrix, optimal gamma, and optimal lambda.
#' @export
cv.gamma <- function(X, Y, tuning_seq = NULL, len_tuning = 60, gamma_seq = NULL, n_gamma = 15, k = 5, id_fold = NULL, eps = 0.001) {
  n <- length(Y)
  
  if (!is.null(gamma_seq)) {
    gamma_seq <- gamma_seq[gamma_seq > 0]
    if (length(gamma_seq) == 0) {
      warning("gamma sequence not supplied or invalid")
      gamma_seq <- NULL
    } else {
      n_gamma <- length(gamma_seq)
    }
  }
  
  if (is.null(gamma_seq)) {
    gamma_seq <- seq(0.1, 3.0, length.out = n_gamma)
  }
  
  # Run initial cv.lambda to get tuning_seq
  fit_cv <- cv.lambda(X, Y, tuning_seq = tuning_seq, len_tuning = len_tuning, gamma = gamma_seq[1], k = k, id_fold = id_fold, eps = eps)
  tuning_seq <- fit_cv$tuning_seq
  len_tuning <- length(tuning_seq)
  
  cv <- matrix(NA, n_gamma, len_tuning)
  
  for (i in 1:n_gamma) {
    cvlamb <- cv.lambda(X, Y, tuning_seq = tuning_seq, len_tuning = len_tuning, gamma = gamma_seq[i], k = k, id_fold = id_fold, eps = eps)
    cv[i, ] <- cvlamb$cv
  }
  
  min_indices <- which(cv == min(cv), arr.ind = TRUE)
  gamma_min <- gamma_seq[min_indices[1, 1]]
  lambda_min <- tuning_seq[min_indices[1, 2]]
  
  return(list(cv = cv, gamma_min = gamma_min, lambda_min = lambda_min))
}
