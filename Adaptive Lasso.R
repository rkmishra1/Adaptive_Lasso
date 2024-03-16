# ADAPTIVE LASSO

# Transforming X by dividing with elements of the weight matrix
# Computing scaled inputs using LARS algorithm
#'  scale_X Scales a matrix of inputs according to  LARS algorithm
#'
#' @param X n x p  design matrix of inputs
#' @param Y n x 1 vector of outputs
#' @param gamma a scalar(>0) input used in the weight(user input)
#'
#' @return A list with the elements
#' \item{X_w}{A n x p matrix scaled according to LARS algorithm}
#' \item{weights}{adaptive weights}
#' @export
#'
#' @examples
#' EXAMPLE
#' X <- matrix(rnorm(500), 50, 10)
#' Y <- rnorm(50)
#' gamma <- 2
#' #Scaling using scale_X
#' sc <- scale_X(X , Y , gamma)
#' #Deriving weighted design matrix
#' X_w <- sc$X_w
#' #Deriving weights
#' weights <- sc$weights
scale_X <- function(X, Y, gamma) {
  glmnet::cv.glmnet(X, Y, alpha = 0)
  Matrix::rankMatrix(X)
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
  # Scales X
  X_w <- t(t(X) / weights)
  # returns scaled X
  return(list(X_w = X_w, weights = weights))
}


# STANDARDIZING INPUTS
#' Standardizes the input design matrix X and output vector Y to mean 0 and scales X
#'@inheritParams scale_X
#'
#' @return
#' \item{Xstd}{scaled X}
#' \item{Ystd}{scaled Y}
#' \item{meanY}{mean of Y}
#' \item{meanX}{Column means after centering the weighted X matrix from scale_X}
#' \item{weights}{weights obtained by centering X_w which is obtained from scale X}
#' @export
#'
#' @examples
#' X <- matrix(rnorm(500), 50, 10)
#' Y <- rnorm(50)
#' gamma <- 2
#' #Standardizing X and Y
#' std <- standardize(X , Y , gamma)
#' #Deriving weighted and centered design matrix
#' Xstd <- std$Xstd
#' #Column means of centered X_w
#' meanX <- std$meanX
#' #Deriving centered Y
#' Ystd <- std$Ystd
#' #Mean of Ystd
#' meanY <- std$meanY
#' # Weights
#' weights <- std$weights
standardize <- function(X, Y, gamma) {
  n <- length(Y)
  p <- ncol(X)
  # Scaling X
  Xstd <- scale_X(X, Y, gamma)$X_w
  meanX <- colMeans(Xstd)
  Xcentered <- scale(Xstd, scale = FALSE)
  weights <- sqrt(diag(crossprod(Xcentered) / n))
  Xstd <- t(t(Xcentered) / weights)
  # Center Y
  meanY <- mean(Y)
  Ystd <- Y - meanY
  # Return:
  # Xstd - centered and appropriately scaled X
  # Ystd - centered Y
  # meanY - the mean of original Y
  # meanX - means of columns of X (vector)
  # weights - weight for scaling X
  return(list(Xstd = Xstd, Ystd = Ystd, meanY = meanY, meanX = meanX, weights = weights))
}


#' Soft-thresholding of a scalar a at level lambda
#'
#' @param a scalar to be soft-thresholded
#' @param lambda level of soft thresholding
#'
#' @return soft-thresholded value
#' @export
#'
#' @examples
#' a = 2
#' lambda = 1
#' soft(a , lambda)
soft <- function(a, lambda) {
  if (a > lambda) {
    return(a - lambda)
  } else if (a < (-lambda)) {
    return(a + lambda)
  } else {
    return(0)
  }
}
# Xstd - Centered and Scaled design matrix of order n x p
# Ystd - centered Y, n x 1
# lamdba - tuning parameter
# beta - value of beta at which to evaluate the function
# Computing the objective function
#'  Function for soft-thresholding
#'
#' @param Xstd n x p design matrix X scaled according to LARS algorithm and centered to mean 0
#' @param Ystd n x 1 centered output vector
#' @param beta   p x 1 vector of parameters
#' @param lambda tuning parameter(scalar)
#'
#' @return Objective function for adaplasso
#' @export
#'
#' @examples
#' X <- matrix(rnorm(500), 50, 10)
#' Y <- rnorm(50)
#' gamma <- 2
#' #Standardizing X and Y
#' std <- standardize(X , Y , gamma)
#' #Deriving weighted and centered design matrix
#' Xstd <- std$Xstd
#' #Deriving centered Y
#' Ystd <- std$Ystd
#' #Defining beta
#' beta <- solve(crossprod(X)) %*% t(X) %*% Y
#' #Lambda value
#' lambda <- 2
#' #Objective function
#' obj <- adaplasso(Xstd, Ystd, beta, lambda)
adaplasso <- function(Xstd, Ystd, beta, lambda) {
  n <- length(Ystd)
  # objective function
  obj <- sum((Ystd - (Xstd %*% beta))^2) / (2 * n) + sum(lambda * abs(beta))
  # Return
  return(obj)
}

# Fit adaptive lasso on standardized data for a given lambda
# Xstd - centered and scaled X, n x p
# Ystd - An input vector of order n x 1
# lamdba - tuning parameter
# beta_init - optional starting point for the coordinate descent algorithm for a given lamba(p x 1 vector)
# eps - precision level for convergence assessment, default 0.001
#' Fits adaptive adaplasso based on standardized data
#'@inheritParams adaplasso
#' @param beta_init p x 1, optional starting point for coordinate descent algorithm
#' @param eps  precision level for convergence assessment, default 0.001
#'
#' @return
#' \item{beta}{vector of parameters}
#' \item{obj_min}{optimal value of the objective function}
#' @export
#'
#' @examples
#' X <- matrix(rnorm(500), 50, 10)
#' Y <- rnorm(50)
#' gamma <- 2
#' #Standardizing X and Y
#' std <- standardize(X , Y , gamma)
#' #Deriving weighted and centered design matrix
#' Xstd <- std$Xstd
#' #Deriving centered Y
#' Ystd <- std$Ystd
#' # tuning parameter
#' lambda  <- 0.1
#' fit <- adaplassostd_lambda(Xstd, Ystd, lambda)
adaplassostd_lambda <- function(Xstd, Ystd, lambda, beta_init = NULL, eps = 0.001) {
  n <- length(Ystd)
  p <- ncol(Xstd)
  # Checking compatibility
  # If Number of rows of X and Dimension of Y match
  if (nrow(Xstd) != length(Ystd)) {
    stop("Error: nrow(X) and length(Y) are not equal")
  }
  # If lambda is non-negative
  if (lambda < 0) {
    stop("Error: Lambda is negative")
  }
  #  Initializing beta_init
  if (is.null(beta_init)) {
    beta_init <- rep(0, p)
  } else if (length(beta_init) != p) {
    stop("Error: dimension of p and ncol(X) do not match", ncol(Xstd))
  }
  #Assigning variables for coordinate descent implementation
  n <- length(Ystd)
  beta <- beta_init
  curr_obj <- adaplasso(Xstd, Ystd, beta, lambda)
  last_obj <- Inf
  r <- Ystd - Xstd %*% beta_init
  while ((last_obj - curr_obj) > eps) {
    for (j in 1:p)
    {
      beta_old <- beta[j]
      beta[j] <- soft(beta[j] + (crossprod(Xstd[, j], r)) / n, lambda)
      r <- r + Xstd[, j] * (beta_old - beta[j])
    }
    
    last_obj <- curr_obj
    curr_obj <- adaplasso(Xstd, Ystd, beta, lambda)
  }
  obj_min <- curr_obj # Minimum value of the objective function
  # Return
  # beta - vector of parameter estimates
  # obj_min - optimal value of the objective function at beta
  return(list(beta = beta, obj_min = obj_min))
}

# Fit adaptive lasso on standardized data for a sequence of lambda values for a given value of gamma.
# Xstd - centered and scaled X, n x p
# Ystd - centered Y, n x 1
# tuning_seq - sequence of tuning parameters, optional
# len_tuning - length of desired tuning parameter sequence
# eps - precision level for convergence assessment, default 0.001
#' Fits Adaptive adaplasso on a sequence of lambda values based on standardized data
#'@inheritParams adaplassostd_lambda
#' @param tuning_seq (optional)sequence of tuning parameters
#' @param len_tuning length of desired tuning parameter sequence
#'
#' @return
#' \item{tuning_seq}{the actual sequence of tuning parameters used}
#' \item{beta_lamb}{matrix of solutions at each lambda value for a given gamma, dimension is p x len_tuning
#' \item{obj_min_vec}{vector of optimal values of the objective function for each lambda at solution}
#' @export
#'
#' @examples
#' X <- matrix(rnorm(500), 50, 10)
#' Y <- rnorm(50)
#' gamma <- 2
#' #Standardizing X and Y
#' std <- standardize(X , Y , gamma)
#' #Deriving weighted and centered design matrix
#' Xstd <- std$Xstd
#' #Deriving centered Y
#' Ystd <- std$Ystd
#' fit <- adaplassostdseq_lambda(Xstd, Ystd)
adaplassostdseq_lambda <- function(Xstd, Ystd, tuning_seq = NULL, len_tuning = 60, eps = 0.001) {
  n <- length(Ystd)
  # Compatibility check for n
  if (nrow(Xstd) != length(Ystd)) {
    stop("Dimensions of X and Y do not match")
  }
  
  # Checks for tuning_seq
  if (is.null(tuning_seq) == FALSE) {
    # If tuning_seq is supplied, only keep values that are >= 0, and make sure the values are sorted from largest to smallest. If none of the supplied values satisfy the requirement, print the warning message and proceed as if the values were not supplied.
    tuning_seq <- sort(tuning_seq[tuning_seq >= 0], decreasing = TRUE)
    if (length(tuning_seq) == 0) {
      print("Warning: Sequence of tuning parameters for fixed gamma not supplied")
      tuning_seq <- NULL
    } else {
      len_tuning <- length(tuning_seq)
    }
  }
  # If tuning_seq is not supplied
  if (is.null(tuning_seq)) {
    lambda_max <- max(abs(crossprod(Xstd, Ystd)) / n)
    tuning_seq <- exp(seq(log(lambda_max), log(0.05), length = len_tuning))
  }
  
  p <- ncol(Xstd)
  beta <- rep(0, p)
  beta_lamb <- matrix(0, p, len_tuning)
  obj_min_vec <- rep(0, len_tuning)
  
  for (i in 1:(len_tuning)) {
    fit <- adaplassostd_lambda(Xstd, Ystd, tuning_seq[i], beta_init = beta, eps)
    beta_lamb[, i] <- fit$beta
    obj_min_vec[i] <- fit$obj_min
    beta <- fit$beta
  }
  
  
  # Output
  # tuning_seq - the actual sequence of tuning parameters used
  # beta_lamb - p x length(tuning_seq) matrix of corresponding solutions at each lambda value
  # obj_min_vec - vector of optimal values of the objective function for each lambda at solution
  return(list(tuning_seq = tuning_seq, beta_lamb = beta_lamb, obj_min_vec = obj_min_vec))
}


#  Fit adaptive lasso on original data using a sequence of lambda values
# X - n x p matrix of covariates
# Y - n x 1 response vector
# tuning_seq - sequence of tuning parameters, optional
# len_tuning - length of desired tuning parameter sequence
# eps - precision level for convergence assessment, default 0.001


#'  Fits adaptive lasso
#'@inheritParams scale_X
#'@inheritParams adaplassostdseq_lambda
#' @return
#' \item{tuning_seq}{the actual sequence of tuning parameters used}
#' \item{beta_lamb}{p x length(tuning_seq) matrix of corresponding solutions at each lambda value (original data without center or scale)}
#' \item{intercept_vec} {Unscaled vector of intercepts for a fixed gamma and for different lambda values}
#' @export
#'
#' @examples
#' #EXAMPLE 1
#' X <- matrix(rnorm(500), 50, 10)
#' Y <- rnorm(50)
#' gamma <- 2
#' # Fits adaptive adaplasso
#' fit <- fitadaplasso(X , Y , gamma = gamma)
#' # EXAMPLE 2
#' X <- matrix(rchisq(500, 3), 50, 10)
#' Y <- rbinom(50)
#' tuning_seq <- runif(100, 1, 2)
#' #Fits adaptive adaplasso
#' fit2 <- fitadaplasso(X, Y, tuning_seq = tuning_seq, gamma = 0.1, eps = 0.002 )
fitadaplasso <- function(X, Y, tuning_seq = NULL, len_tuning = 60, gamma = 0.01, eps = 0.001) {
  # Center and standardize X,Y based on standardize and scale_X functions
  sc <- scale_X(X, Y, gamma)
  Std <- standardize(X, Y, gamma)
  X <- Std$Xstd
  Y <- Std$Ystd
  fit <- adaplassostdseq_lambda(X, Y, tuning_seq, len_tuning, eps)
  len_tuning <- length(fit$tuning_seq)
  tuning_seq <- fit$tuning_seq
  # Scaling and centering to get original intercept and coefficient vector for each lambda
  beta <- fit$beta_lamb
  beta_lamb <- beta / (sc$weights * Std$weights)
  intercept_vec <- Std$meanY - ((Std$meanX) %*% beta_lamb)
  
  # output
  # tuning_seq - the actual sequence of tuning parameters used
  # beta_lamb - p x length(tuning_seq) matrix of corresponding solutions at each lambda value (original data without center or scale)
  # intercept_vec - Unscaled vector of intercepts for a fixed gamma and for different lambda values
  return(list(tuning_seq = tuning_seq, beta_lamb = beta_lamb, intercept_vec = intercept_vec))
}




#' Perform cross-validation to select the best fit and finds the optimal lambda for a particular gamma value
#' @inheritParams fitadaplasso
#' @param k  number of folds for k-fold cross-validation, default is 5
#' @param id_fold (optional) vector of length n specifying the folds assignment (from 1 to max(folds_ids)), if supplied the value of k is ignored
#'
#'
#' @return
#' \item{tuning_seq}{the actual sequence of tuning parameters used}
#' \item{beta_lamb}{p x length(tuning_seq) matrix of corresponding solutions at each lambda value (original data without center or scale)}
#' \item{intercept_vec}{Unscaled vector of intercepts for a fixed gamma and for different lambda values}
#' \item{id_fold}{used splitting into folds from 1 to k (either as supplied or as generated in the beginning)}
#' \item{lambda_min}{selected lambda based on minimal rule}
#' \item{cv}{values of CV(lambda) for each lambda}
#' @export
#'
#' @examples
#' X <- matrix(rnorm(500), 50, 10)
#' Y <- rnorm(50)
#' fit_cv <- cv.lambda(X, Y)
cv.lambda <- function(X, Y, tuning_seq = NULL, len_tuning = 60, gamma = 0.01, k = 5, id_fold = NULL, eps = 0.001) {
  n <- length(Y)
  # Fit adaptive lasso on original data using fitadaplasso
  fit_adaplasso <- fitadaplasso(X, Y, tuning_seq, len_tuning, eps)
  # Splitting data according to fold ids
  if (is.null(id_fold)) {
    id_fold <- sample(1:n, size = n) %% k + 1
  }
  
  
  # Defining vectors for loop over folds and lambdas
  tuning_seq <- fit_adaplasso$tuning_seq
  len_tuning <- length(tuning_seq)
  cv <- rep(NA, len_tuning) # want to have CV(lambda)
  cvse <- rep(NA, len_tuning) # want to have SE_CV(lambda)
  cv_folds <- matrix(NA, k, len_tuning)
  
  for (fold in 1:k) {
    #  training data xtrain and ytrain
    Xtrain <- X[id_fold != fold, ]
    Ytrain <- Y[id_fold != fold]
    
    
    # testing data xtest and ytest
    Xtest <- X[id_fold == fold, ]
    Ytest <- Y[id_fold == fold]
    
    # Fitting adaptive lasso
    adaplasso <- fitadaplasso(Xtrain, Ytrain, tuning_seq, gamma, len_tuning, eps)
    cv_folds[fold, ] <- colSums((Ytest - t(c(adaplasso$intercept_vec) + t(Xtest %*% adaplasso$beta_lamb)))^2)
  }
  
  
  # To get cv and cvse from cv_folds
  beta_lamb <- fit_adaplasso$beta_lamb
  intercept_vec <- fit_adaplasso$intercept_vec
  cv <- colMeans(cv_folds)
  cvse <- apply(cv_folds, 2, sd) / sqrt(k)
  
  # Optimal lambda
  min <- which.min(cv)
  lambda_min <- tuning_seq[min]
  # Output
  # tuning_seq - the actual sequence of tuning parameters used
  # beta_lamb - p x length(tuning_seq) matrix of corresponding solutions at each lambda value (original data without center or scale)
  # intercept_vec - Unscaled vector of intercepts for a fixed gamma and for different lambda values
  # id_fold - fold splits
  # lambda_min - optimal solution for lambda
  # cv - values of CV(lambda) for each lambda for a fixed gamma
  return(list(tuning_seq = tuning_seq, beta_lamb = beta_lamb, intercept_vec = intercept_vec, id_fold = id_fold, lambda_min = lambda_min, cv = cv))
}
# Cross-Validation to choose gamma from a sequence of gamma values
#' Cross-Validation to choose the optimal gamma from a sequence of gamma values for a particular sequence of lambdas
#'@inheritParams cv.lambda
#' @param gamma_seq (optional)sequence of gamma values(used in determining weights)
#' @param n_gamma length of the desired sequence of gamma values

#' @return
#' \item{cv}{a n_gamma x len_tuning matrix giving CV(lambda, gamma) for each pair of (lambda, gamma)}
#' \item{gamma_min}{optimal gamma}
#' \item{lambda_min}{selected lambda based on minimal rule}
#' @export
#'
#' @examples
#' X <- matrix(rnorm(500), 50, 10)
#' Y <- rnorm(50)
#' fit_cv_gamma <- cv.gamma(X , Y)
cv.gamma <- function(X, Y, tuning_seq = NULL, len_tuning = 60, gamma_seq = NULL, n_gamma = 60, k = 5, id_fold = NULL, eps = 0.001) {
  n <- length(Y)
  #  Check for the user-supplied gamma-seq
  if (is.null(tuning_seq) == FALSE) {
    # If gamma_seq is supplied, only keep values that are >= 0, and make sure the values are sorted from largest to smallest. If none of the supplied values satisfy the requirement, print the warning message and proceed as if the values were not supplied.
    gamma_seq <- gamma_seq[gamma_seq > 0]
    if (length(gamma_seq) == 0) {
      print("Warning: gamma sequence not supplied")
      gamma_seq <- NULL
    } else {
      n_gamma <- length(gamma_seq)
    }
  }
  #If gamma_seq is not supplied
  if (is.null(gamma_seq)) {
    gamma_seq <- seq(0.0001, 10, by = 0.1)
    n_gamma <- length(gamma_seq)
  }
  fit_cv <- cv.lambda(X, Y, tuning_seq , len_tuning , gamma , k , id_fold , eps)
  tuning_seq <- fit_cv$tuning_seq
  # defining a cross-validation matrix
  cv <- matrix(NA, n_gamma, len_tuning)
  
  for (i in 1:n_gamma) {
    cvlamb <- cv.lambda(X, Y, tuning_seq, len_tuning, gamma_seq[i], k, id_fold, eps)
    cv[i, ] <- cvlamb$cv
  }
  #Finds the row corresponding to the minimum entry of the matrix
  gamma_min_ind <- which(cv == min(cv), arr.ind = T)[1]
  #Finds the gamma which minimizes the cross-validation error
  gamma_min <- gamma_seq[gamma_min_ind]
  #Finds the column corresponding to the minimum entry of the matrix
  lambda_min_ind <- which(cv == min(cv), arr.ind = T)[2]
  #Finds the gamma which minimizes the cross-validation error
  lambda_min <- tuning_seq[lambda_min_ind]
  
  # Return
  return(list(cv = cv, gamma_min = gamma_min,lambda_min = lambda_min))
}