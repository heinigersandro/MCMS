# Helper functions

get_p_value=function(M, L, X, H, Z, u, v, B, b, num_B_cov, to_add_ID, mask, permutation_max, permutation_inference){
  # get dimensions of data
  n_col=ncol(M)
  n_row=nrow(M)
  
  eval_Bb= matrix(0, n_row, n_col)
  for(i in 1:n_col){
    eval_Bb[,i]=B[, ((i-1)*num_B_cov+1):(i*num_B_cov)] %*% b
  }
  
  if(to_add_ID){
    X=cbind(X,diag(n_row))
    Z=rbind(Z,diag(n_col))
  }
  
  # Calculate residuals
  eps = M - 
    (L +
       X %*% H %*% Z + 
       u %*% t(rep(1,n_col)) + 
       rep(1,n_row) %*% t(v) + 
       eval_Bb)
  
  # Pre-calculations for loop
  n_permutations=min(permutation_max,factorial(n_col), 10000)
  permutations=vector(mode="numeric", length=n_permutations)
  mask_new=(1-mask)
  n_mask_new=sum(mask_new)
  
  # Determine value of estimator
  estimator=1/n_mask_new*sum(abs(mask_new*eps))
  
  # Generate random permutations of residuals
  for(i in 1:n_permutations){
    if(permutation_inference=="block_moving") {
      eps_i=t(apply(eps,1, function(x) {shifter=floor(runif(1)*n_col); x[(1:n_col+shifter) %% n_col +1]}))
    } else {
      eps_i=eps[,sample(1:ncol(M))]
    }
    permutations[i]=1/n_mask_new*sum(abs(mask_new*eps_i))
  }
  p_value=mean(permutations>estimator)
}

validate_input_parameters=function(M, 
                                   mask, 
                                   X = NA, 
                                   Z = NA, 
                                   B = NA, 
                                   num_B_cov = NA, 
                                   to_add_ID = FALSE, 
                                   to_estimate_u = FALSE, 
                                   to_estimate_v = FALSE, 
                                   to_estimate_b = FALSE, 
                                   to_estimate_H = FALSE, 
                                   niter = 100, 
                                   rel_tol = 1e-5, 
                                   to_normalize = FALSE, 
                                   cv_ratio = NA,
                                   cv_criterion = "1se",
                                   num_folds = 3,
                                   n_config = 150, 
                                   cube_search = FALSE, 
                                   is_quiet = FALSE, 
                                   post_estimation = FALSE, 
                                   impose_null = FALSE,
                                   permutation_inference = "all",
                                   permutation_max = 1000,
                                   model_selection_H = FALSE, 
                                   model_selection_b = FALSE,
                                   seed = NA,
                                   lambda_L = 0,
                                   lambda_H = 0,
                                   lambda_b = 0,
                                   n_lambda = 50){
  # check all inputs
  if(!is.matrix(M) | !is.numeric(M) ){ stop( "M has to be a numeric matrix")}
  if(!is.matrix(mask) | !is.numeric(mask) ){ stop( "mask has to be a numeric matrix")}
  if(!(isNA(X) | (is.matrix(X) & is.numeric(X))) ){ stop( "X has to be a numeric matrix")}
  if(!(isNA(Z) | (is.matrix(Z) & is.numeric(Z))) ){ stop( "Z has to be a numeric matrix")}
  if(!(isNA(B) | (is.matrix(B) & is.numeric(B))) ){ stop( "B has to be a numeric matrix")}
  if(!(isNA(num_B_cov) | is.numeric(num_B_cov)) ){ stop( "num_B_cov has to be an integer")}
  if(!is.logical(to_add_ID)){ stop( "to_add_ID has to be a logical")}
  if(!is.logical(to_estimate_u)){ stop( "to_estimate_u has to be a logical")}
  if(!is.logical(to_estimate_v)){ stop( "to_estimate_v has to be a logical")}
  if(!is.logical(to_estimate_b)){ stop( "to_estimate_b has to be a logical")}
  if(!is.logical(to_estimate_H)){ stop( "to_estimate_H has to be a logical")}
  if(!is.numeric(niter) | niter<1){ stop( "niter has to be a positive integer")}
  if(!is.numeric(rel_tol) ){ stop( "rel_tol has to be numeric")}
  if(rel_tol<0 | rel_tol>0.5){ warning("Value of rel_tol is unreasonably chosen. Should be a small, positive value")}
  if(!is.logical(to_normalize)){ stop( "to_normalize has to be a logical")}
  if(!(isNA(cv_ratio) | is.numeric(cv_ratio)) ){ stop( "cv_ratio has to be numeric")}
  if(!isNA(cv_ratio)){if(cv_ratio<0.5 | rel_tol>=1){ warning("Value of cv_ratio is unreasonably chosen. Should be in [0.5,1)")}}
  if(!(cv_criterion %in% c("mse","1se","both"))){ stop('cv_criterion should be one of c("mse","1se","both")')}
  if(!is.numeric(num_folds) | num_folds<1){ stop( "num_folds has to be a positive integer")}
  if(num_folds>10){ warning("num_folds is large. Estimation may take a long time.")}
  if(!is.numeric(n_config) | n_config<1){ stop( "n_config has to be a positive integer")}
  if(n_config>250){ warning("n_config is large. Estimation may take a long time.")}
  if(n_config<20){ warning("n_config is small. Iteration procedure may not converge.")}
  if(!is.logical(cube_search)){ stop( "cube_search has to be a logical")}
  if(!is.logical(is_quiet)){ stop( "is_quiet has to be a logical")}
  if(!is.logical(post_estimation)){ stop( "post_estimation has to be a logical")}
  if(!is.logical(impose_null)){ stop( "impose_null has to be a logical")}
  if(!(permutation_inference %in% c("all","block_moving"))){ stop('permutation_inference should be one of c("all","block_moving")')}
  if(!is.numeric(permutation_max) | permutation_max<100){ stop( "permutation_max has to be an integer largen than 100")}
  if(permutation_max>10000){ warning("permutation_max is large. Estimation may take a long time.")}
  if(!is.logical(model_selection_H)){ stop( "model_selection_H has to be a logical")}
  if(!is.logical(model_selection_b)){ stop( "model_selection_b has to be a logical")}
  if(!(isNA(seed) | is.numeric(seed)) ){ stop( "seed has to be numeric")}
  if(!is.numeric(lambda_L) | lambda_L<0){ stop( "lambda_L has to be >=0")}
  if(!is.numeric(lambda_H) | lambda_L<0){ stop( "lambda_H has to be >=0")}
  if(!is.numeric(lambda_b) | lambda_L<0){ stop( "lambda_b has to be >=0")}
  if(!is.numeric(n_lambda) | n_lambda<1){ stop( "n_lambda should be a positive integer")}
  
  # Check applicability of parameters
  if(isNA(num_B_cov) & !isNA(B)){ stop("Please specify the number of covariates in B.") }
  if(!isNA(B)){if(ncol(B)/ncol(M)!=num_B_cov){ stop("Specified num_B_cov does not match matrix dimensions.") }}
  
  zero_tol=1e-8
  if(!isNA(X)){if(min(apply(X,2,sd))<zero_tol){ stop("Constant variable in X covariates.") }}
  if(!isNA(Z)){if(min(apply(Z,1,sd))<zero_tol){ stop("Constant variable in Z covariates.") }}
  if(!isNA(B)){
    for(i in 0:(num_B_cov-1)){
      if(sd(B[,which(1:ncol(B) %% num_B_cov == i)])<zero_tol){ stop("Constant variable in Z covariates.") }
    }
  }
  
  if(model_selection_H & isNA(X) & isNA(Z)){ stop("Specify any covariate matrix (X or Z), or disable H regularization (model_selection_H).") }
  if(model_selection_b & isNA(B)){ stop("Specify unit/time-specific covariate matrix B or disable b regularization (model_selection_b).") }
  
  if(to_estimate_H & isNA(X) & isNA(Z)){ stop("Specify any covariate matrix (X or Z), or disable H estimation (to_estimate_H).") }
  if(to_estimate_b & isNA(B)){ stop("Specify unit/time-specific covariate matrix B or disable b estimation (to_estimate_b).") }

  if(model_selection_H & !to_estimate_H){ stop("covariate regularisation in H is only possible if to_estimate_H=TRUE")}
  if(model_selection_b & !to_estimate_b){ stop("covariate regularisation in H is only possible if to_estimate_H=TRUE")}
}

#' This function is used to estimate the maximum lambda_L, lambda_H, and lambda_b values for the MCNNM algorithm.
#'
#' @param M Outcome matrix
#' @param mask Treatment mask matrix (1 if control, 0 if treated)
#' @param X Units covariate matrix (Time-invariant) (default: NA)
#' @param Z Time covariate matrix (Unit-invariant) (default: NA)
#' @param B Unit-Time covariate matrix (default: NA)
#' @param num_B_cov Number of unit-time covariates (default: NA)
#' @param to_add_ID Whether to add an identity matrix to the X and Z matrices. (default: TRUE)
#' @param to_estimate_u Whether to estimate the units' fixed effects. (default: TRUE)
#' @param to_estimate_v Whether to estimate the time's fixed effects. (default: TRUE)
#' @param niter Number of iterations for the coordinate descent steps. (default: 100)
#' @param rel_tol Relative tolerance for the convergence of the coordinate descent steps. (default: 1e-5)
#' @param to_normalize Whether to normalize the X, Z, and B matrices. (default: TRUE)
#' @param model_selection_H Whether the unit and time varying covariate link matrix should be regularized. (default: TRUE)
#' @param model_selection_b Whether the unit-time varying covariates parameters should be regularized. (default: TRUE)
#' @param is_quiet Whether to print the progress of the algorithm. (default: FALSE)
#' @return (List) List of the maximum lambda_L, lambda_H, and lambda_b values
#' @examples 
#' mcms_lam_range(M, X, Z, B, mask, num_B_cov)
mcms_lam_range=function(M, 
                        mask,
                        X = NA, 
                        Z = NA, 
                        B = NA, 
                        num_B_cov = NA, 
                        to_add_ID = TRUE, 
                        to_estimate_u = TRUE, 
                        to_estimate_v = TRUE,
                        niter = 150,
                        rel_tol = 1e-5, 
                        to_normalize = TRUE, 
                        model_selection_H = TRUE, 
                        model_selection_b = TRUE, 
                        is_quiet = FALSE){
  start_time=Sys.time()
  
  validate_input_parameters(M, 
                            mask, 
                            X = X, 
                            Z = Z, 
                            B = B, 
                            num_B_cov = num_B_cov, 
                            to_add_ID = to_add_ID, 
                            to_estimate_u = to_estimate_u, 
                            to_estimate_v = to_estimate_v,
                            to_estimate_H = model_selection_H,
                            to_estimate_b = model_selection_b,
                            niter = niter, 
                            rel_tol = rel_tol, 
                            to_normalize = to_normalize, 
                            is_quiet = is_quiet, 
                            model_selection_H = model_selection_H, 
                            model_selection_b = model_selection_b)
  
  # Set default covariate matrices for seamless estimation in case they are not specified by the user
  if(isNA(X)){X=matrix(data=1, nrow=nrow(M), ncol=1)}
  if(isNA(Z)){Z=matrix(data=1, nrow=1, ncol=ncol(M))}
  if(isNA(B)){B=matrix(data=0, nrow=nrow(M), ncol=ncol(M))}
  
  cpp_output=mc_ms_lam_range(M,
                             X,
                             t(Z),
                             B,
                             mask,
                             num_B_cov, 
                             to_add_ID, 
                             to_estimate_u, 
                             to_estimate_v,
                             niter,
                             rel_tol,
                             to_normalize,
                             is_quiet)
  # remove max lamba values if model selection is not selected
  if(!model_selection_H){
    cpp_output=cpp_output[names(cpp_output) != "lambda_H_max"]
  }
  if(!model_selection_b){
    cpp_output=cpp_output[names(cpp_output) != "lambda_b_max"]
  }
  
  duration <-  abs(as.numeric(difftime(Sys.time(), start_time, units="secs")))
  if(!is_quiet){
    if(duration>86400){
      cat(sprintf("Runtime: %02d days, %02d hours,  %02d mins, %02d seconds\n", duration %/% 86400, duration %% 86400 %/% 3600, duration %% 3600 %/% 60, duration %% 60 %/% 1))	
    } else if(duration>3600){
      cat(sprintf("Runtime: %02d hours,  %02d mins, %02d seconds\n", duration %/% 3600, duration %% 3600 %/% 60, duration %% 60 %/% 1))	
    } else if(duration>60){
      cat(sprintf("Runtime: %02d mins, %02d seconds\n", duration %/% 60, duration %% 60 %/% 1))	
    } else {
      cat(sprintf("Runtime: %02d seconds\n", duration %/% 1))	
    }
  }
  return(cpp_output)
}

#' This function is used to estimate the MCNNM model with given lambda values for L,H, and b.
#' If post_estimation is set, then the estimated parameters are returned additionally in the list.
#' 
#' @param M Outcome matrix
#' @param mask Treatment mask matrix (1 if control, 0 if treated)
#' @param lambda_L lambda value for rank-regularization
#' @param lambda_H lambda value for unit and time covariate linking matrix regularization
#' @param lambda_b lambda value for unit-time covariate regularization
#' @param X Units covariate matrix (Time-invariant)
#' @param Z Time covariate matrix (Unit-invariant)
#' @param B Unit-Time covariate matrix
#' @param num_B_cov Number of unit-time covariates
#' @param to_add_ID Whether to add an identity matrix to the X and Z matrices. (default: TRUE)
#' @param to_estimate_u Whether to estimate the units' fixed effects. (default: TRUE)
#' @param to_estimate_v Whether to estimate the time's fixed effects. (default: TRUE)
#' @param to_estimate_b Whether to estimate the unit-time varying covariate parameters. (default: TRUE)
#' @param to_estimate_H Whether to estimate the unit and time covariates linking parameters. (default: TRUE)
#' @param niter Number of iterations for the coordinate descent steps. (default: 100)
#' @param rel_tol Relative tolerance for the convergence of the coordinate descent steps. (default: 1e-5)
#' @param to_normalize Whether to normalize the X, Z, and B matrices. (default: TRUE)
#' @param is_quiet Whether to print the progress of the algorithm. (default: FALSE)
#' @param post_estimation Whether a post regularization estimation without regularization should be estimated
#' @param impose_null Whether null-hypothesis should be imposed. Implies that all observations also treated observations are used in model estimation. (default: TRUE)
#' @param permutation_inference Permutation setting for inference procedure. Either "block-moving" under stationarity assumption, "all" under iid assumption or NA if no inference procedure should be executed. Is only possible if impose_null is set to TRUE. (default: "all")
#' @param permutation_max Maximum of evaluated permutations. If NA, all possible permutations are considered (default: NA)
#' @return: List containing the following elements:
#' 	    - "u": Estimated units' fixed effects
#' 		- "v": Estimated time's fixed effects
#' 		- "B": Estimated unit-time covariates' coefficients
#' 		- "H": Estimated H matrix
#' 		- "tau": The estimate of the treatment effect
#' 		- "tau_rot": Rule-of-thumb corrected estimate of the treatment effect
#' 		- "lambda_L": Estimated lambda_L value
#' 		- "lambda_H": Estimated lambda_H value
#' 		- "lambda_b": Estimated lambda_b value
#' @examples 
#' mcms_fit(M, X, Z, B, mask, lambda_L, lambda_H, lambda_b, num_B_cov)

mcms_fit=function(M, 
                  mask,
                  lambda_L, 
                  lambda_H = 0, 
                  lambda_b = 0, 
                  X = NA, 
                  Z = NA, 
                  B = NA, 
                  num_B_cov = NA, 
                  to_add_ID = TRUE, 
                  to_estimate_u = TRUE, 
                  to_estimate_v = TRUE, 
                  to_estimate_b = TRUE, 
                  to_estimate_H = TRUE, 
                  niter = 100, 
                  rel_tol = 1e-5, 
                  to_normalize = TRUE, 
                  is_quiet = FALSE, 
                  post_estimation = TRUE, 
                  impose_null = TRUE,
                  permutation_inference = "all",
                  permutation_max = 1000){
  start_time=Sys.time()
  
  # Check input parameters (additional checks are performed in the cpp code)
  validate_input_parameters(M = M, 
                            mask = mask, 
                            X = X, 
                            Z = Z, 
                            B = B, 
                            num_B_cov = num_B_cov, 
                            to_add_ID = to_add_ID, 
                            to_estimate_u = to_estimate_u, 
                            to_estimate_v = to_estimate_v, 
                            to_estimate_b = to_estimate_b, 
                            to_estimate_H = to_estimate_H, 
                            niter = niter, 
                            rel_tol = rel_tol, 
                            to_normalize = to_normalize, 
                            is_quiet = is_quiet, 
                            post_estimation = post_estimation, 
                            impose_null = impose_null,
                            permutation_inference = permutation_inference,
                            permutation_max = permutation_max,
                            lambda_L=lambda_L,
                            lambda_H=lambda_H,
                            lambda_b=lambda_b)
  
  # Set default covariate matrices for seamless estimation in case they are not specified by the user
  if(isNA(X)){X=matrix(data=1, nrow=nrow(M), ncol=1)}
  if(isNA(Z)){Z=matrix(data=1, nrow=1, ncol=ncol(M))}
  if(isNA(B)){B=matrix(data=0, nrow=nrow(M), ncol=ncol(M))}
  model_selection_H = lambda_H>0
  model_selection_b = lambda_b>0
  
  # call the cpp function
  cpp_output=mc_ms_fit(M, 
                       X, 
                       t(Z), 
                       B, 
                       mask, 
                       lambda_L, 
                       lambda_H, 
                       lambda_b, 
                       num_B_cov, 
                       to_add_ID, 
                       to_estimate_u, 
                       to_estimate_v, 
                       to_estimate_b, 
                       to_estimate_H, 
                       niter, 
                       rel_tol, 
                       to_normalize, 
                       is_quiet, 
                       post_estimation, 
                       impose_null,
                       model_selection_H,
                       model_selection_b)
  
  # calculate the p-value
  if(!isNA(permutation_inference)){
    cpp_output[["p_value"]]=get_p_value(M=M, 
                                        L=cpp_output[["L"]], 
                                        X=X, 
                                        H=cpp_output[["H"]], 
                                        Z=Z, 
                                        u = cpp_output[["u"]], 
                                        v = cpp_output[["v"]], 
                                        B = B, 
                                        b = cpp_output[["b"]], 
                                        num_B_cov = num_B_cov, 
                                        to_add_ID = to_add_ID,
                                        mask = mask, 
                                        permutation_max= permutation_max, 
                                        permutation_inference = permutation_inference)
  }
  
  # apply rule of thumb correction for treatment effect under imposed null
  if(impose_null){
    cpp_output[["tau_rot"]]=cpp_output[["tau"]]/mean(mask)
  }
  
  duration <-  abs(as.numeric(difftime(Sys.time(), start_time, units="secs")))
  if(!is_quiet){
    if(duration>86400){
      cat(sprintf("Runtime: %02d days, %02d hours,  %02d mins, %02d seconds\n", duration %/% 86400, duration %% 86400 %/% 3600, duration %% 3600 %/% 60, duration %% 60 %/% 1))	
    } else if(duration>3600){
      cat(sprintf("Runtime: %02d hours,  %02d mins, %02d seconds\n", duration %/% 3600, duration %% 3600 %/% 60, duration %% 60 %/% 1))	
    } else if(duration>60){
      cat(sprintf("Runtime: %02d mins, %02d seconds\n", duration %/% 60, duration %% 60 %/% 1))	
    } else {
      cat(sprintf("Runtime: %02d seconds\n", duration %/% 1))	
    } 
  }
  
  return(cpp_output)
}

#' This function performs cross-validation to estimate the optimal lambda values and runs the mcnnm_wc model.	
#' If post_estimation is set, then the estimated parameters are returned additionally in the list.
#' 
#' @param M Outcome matrix
#' @param mask Treatment mask matrix (1 if control, 0 if treated)
#' @param X Units covariate matrix (Time-invariant) (default: NA)
#' @param Z Time covariate matrix (Unit-invariant) (default: NA)
#' @param B Unit-Time covariate matrix (default: NA)
#' @param num_B_cov Number of unit-time covariates (default: NA)
#' @param to_add_ID Whether to add an identity matrix to the X and Z matrices. (default: TRUE)
#' @param to_estimate_u Whether to estimate the units' fixed effects. (default: TRUE)
#' @param to_estimate_v Whether to estimate the time's fixed effects. (default: TRUE)
#' @param to_estimate_b Whether to estimate the unit-time varying covariate parameters. (default: TRUE)
#' @param to_estimate_H Whether to estimate the unit and time covariates linking parameters. (default: TRUE)
#' @param niter Number of iterations for the coordinate descent steps. (default: 100)
#' @param rel_tol Relative tolerance for the convergence of the coordinate descent steps. (default: 1e-5)
#' @param to_normalize Whether to normalize the X, Z, and B matrices. (default: TRUE)
#' @param cv_ratio Share of observations that is used for training in cross-validation. If NA is provided, then the share is equal to the share of non-treated observations in the sample (default: NA)
#' @param cv_criterion One of c("mse","1.se","both"). Optimality criterion for cross-validation. "mse" selects the optimal lambda configuration based on mse, "1.se" the largest lambda configuration for which the error is lower than the minimal mse + one standards error at optimal level. "both" returns both estimates. (Default: "mse") 
#' @param num_folds Number of folds in cross-validation (default: 3)
#' @param n_config Number of configurations of lambda that are evaluated in cross-validation (default: 150)
#' @param cube_search Search method for lambda configurations. (default: TRUE)
#' If set to true, then configurations are evaluated in a hypercube fashion with zooming in on best edge every iteration
#' If set to false, lambda values are spaned on a grid.
#' @param is_quiet Whether to print the progress of the algorithm. (default: FALSE)
#' @param post_estimation Whether a post regularization estimation without regularization should be estimated. (default: TRUE)
#' @param impose_null Whether null-hypothesis should be imposed. Implies that all observations also treated observations are used in model estimation. (default: TRUE)
#' @param permutation_inference Permutation setting for inference procedure. Either "block-moving" under stationarity assumption, "all" under iid assumption or NA if no inference procedure should be executed. Is only possible if impose_null is set to TRUE. (default: "all")
#' @param permutation_max Maximum of evaluated permutations. If NA, all possible permutations are considered (default: NA)
#' @param model_selection_H Whether the unit and time varying covariate link matrix should be regularized. (default: TRUE)
#' @param model_selection_b Whether the unit-time varying covariates parameters should be regularized. (default: TRUE)
#' @param seed Any integer to control the random seed. NA for a random seed (default: NA)
#' @return: List containing the following elements:
#' 	    - "u": Estimated units' fixed effects
#' 		- "v": Estimated time's fixed effects
#' 		- "B": Estimated unit-time covariates' coefficients
#' 		- "H": Estimated H matrix
#' 		- "tau": The estimate treatment effect
#' 		- "tau_rot": The rule-of-thumb corrected estimate of the treatment effect
#' 		- "lambda_L": Estimated lambda_L value
#' 		- "lambda_H": Estimated lambda_H value
#' 		- "lambda_b": Estimated lambda_b value
#' @examples 
#' mcms_cv(M, X, Z, B, mask, num_B_cov)
#' 
mcms_cv=function(M, 
                 mask, 
                 X = NA, 
                 Z = NA, 
                 B = NA, 
                 num_B_cov = NA, 
                 to_add_ID = TRUE, 
                 to_estimate_u = TRUE, 
                 to_estimate_v = TRUE, 
                 to_estimate_b = TRUE, 
                 to_estimate_H = TRUE, 
                 niter = 100, 
                 rel_tol = 1e-5, 
                 to_normalize = TRUE, 
                 cv_ratio = NA,
                 cv_criterion = "1se",
                 num_folds = 3,
                 n_config = 150, 
                 cube_search = TRUE, 
                 is_quiet = TRUE, 
                 post_estimation = TRUE, 
                 impose_null = TRUE,
                 permutation_inference = "all",
                 permutation_max = 1000,
                 model_selection_H = TRUE, 
                 model_selection_b = TRUE,
                 seed = NA){
  start_time=Sys.time()
  
  # Check input parameters (additional checks are performed in the cpp code)
  validate_input_parameters(M = M, 
                            mask = mask, 
                            X = X, 
                            Z = Z, 
                            B = B, 
                            num_B_cov = num_B_cov, 
                            to_add_ID = to_add_ID, 
                            to_estimate_u = to_estimate_u, 
                            to_estimate_v = to_estimate_v, 
                            to_estimate_b = to_estimate_b, 
                            to_estimate_H = to_estimate_H, 
                            niter = niter, 
                            rel_tol = rel_tol, 
                            to_normalize = to_normalize, 
                            cv_ratio = cv_ratio,
                            cv_criterion = cv_criterion,
                            num_folds = num_folds,
                            n_config = n_config, 
                            cube_search = cube_search, 
                            is_quiet = is_quiet, 
                            post_estimation = post_estimation, 
                            impose_null = impose_null,
                            permutation_inference = permutation_inference,
                            permutation_max = permutation_max,
                            model_selection_H = model_selection_H, 
                            model_selection_b = model_selection_b,
                            seed = seed)
  
  # Set estimation parameters in case they have not been specified by the user
  
  # Set default covariate matrices for seamless estimation in case they are not specified by the user
  if(isNA(X)){X=matrix(data=1, nrow=nrow(M), ncol=1)}
  if(isNA(Z)){Z=matrix(data=1, nrow=1, ncol=ncol(M))}
  if(isNA(B)){B=matrix(data=0, nrow=nrow(M), ncol=ncol(M))}
  
  # set a random seed if none is specified
  if(isNA(seed)){
    seed=sample.int(99999,1)
  }
  
  # take share of not-treated observations
  if(isNA(cv_ratio)){
    cv_ratio=mean(mask)
  }
  
  # Which CV criterion should be applied
  if(cv_criterion %in% c("mse","both")){
    return_mse=T
  } else {return_mse=F}
  if(cv_criterion %in% c("1.se","both")){
    return_1se=T
  } else {return_1se=F}
  
  # call cpp function
  cpp_output=mc_ms_cv(M, 
                      X,
                      t(Z),
                      B,
                      mask,
                      num_B_cov, 
                      to_add_ID, 
                      to_estimate_u, 
                      to_estimate_v, 
                      to_estimate_b, 
                      to_estimate_H, 
                      niter, 
                      rel_tol, 
                      to_normalize, 
                      cv_ratio,
                      num_folds,
                      n_config, 
                      cube_search, 
                      is_quiet, 
                      post_estimation, 
                      impose_null,
                      model_selection_H, 
                      model_selection_b,
                      return_mse=return_mse,
                      return_1se=return_1se,
                      seed)
  
  # calculate the p-value
  if((!isNA(permutation_inference)) & cv_criterion %in% c("mse","both")){
    cpp_output[["p_value"]]=get_p_value(M=M,
                                        L=cpp_output[["L"]],
                                        X=X,
                                        H=cpp_output[["H"]],
                                        Z=Z,
                                        u = cpp_output[["u"]],
                                        v = cpp_output[["v"]],
                                        B = B,
                                        b = cpp_output[["b"]],
                                        num_B_cov = num_B_cov,
                                        to_add_ID = to_add_ID,
                                        mask = mask,
                                        permutation_max= permutation_max,
                                        permutation_inference = permutation_inference)
  }
  
  if((!isNA(permutation_inference)) & cv_criterion %in% c("1.se","both")){
    cpp_output[["p_value_1se"]]=get_p_value(M=M,
                                            L=cpp_output[["L_1se"]],
                                            X=X,
                                            H=cpp_output[["H_1se"]],
                                            Z=Z,
                                            u = cpp_output[["u_1se"]],
                                            v = cpp_output[["v_1se"]],
                                            B = B,
                                            b = cpp_output[["b_1se"]],
                                            num_B_cov = num_B_cov,
                                            to_add_ID = to_add_ID,
                                            mask = mask,
                                            permutation_max= permutation_max,
                                            permutation_inference = permutation_inference)
  }
  
  # apply rule of thumb correction for treatment effect under imposed null
  if(impose_null & cv_criterion %in% c("mse","both")){
    cpp_output[["tau_rot"]]=cpp_output[["tau"]]/mean(mask)
  }
  if(impose_null & cv_criterion %in% c("1.se","both")){
    cpp_output[["tau_1se_rot"]]=cpp_output[["tau_1se"]]/mean(mask)
  }
  
  if(!is_quiet){
    duration <-  abs(as.numeric(difftime(Sys.time(), start_time, units="secs")))
    if(duration>86400){
      cat(sprintf("Runtime: %02d days, %02d hours,  %02d mins, %02d seconds\n", duration %/% 86400, duration %% 86400 %/% 3600, duration %% 3600 %/% 60, duration %% 60 %/% 1))	
    } else if(duration>3600){
      cat(sprintf("Runtime: %02d hours,  %02d mins, %02d seconds\n", duration %/% 3600, duration %% 3600 %/% 60, duration %% 60 %/% 1))	
    } else if(duration>60){
      cat(sprintf("Runtime: %02d mins, %02d seconds\n", duration %/% 60, duration %% 60 %/% 1))	
    } else {
      cat(sprintf("Runtime: %02d seconds\n", duration %/% 1))	
    } 
  }
  
  return(cpp_output)
}
#' This function estimate deviations from optimal lambda
#' If post_estimation is set, then the estimated parameters are returned additionally in the list.
#' 
#' @param M Outcome matrix
#' @param mask Treatment mask matrix (1 if control, 0 if treated)
#' @param X Units covariate matrix (Time-invariant) (default: NA)
#' @param Z Time covariate matrix (Unit-invariant) (default: NA)
#' @param B Unit-Time covariate matrix (default: NA)
#' @param num_B_cov Number of unit-time covariates (default: NA)
#' @param to_normalize Whether to normalize the X, Z, and B matrices. (default: TRUE)
#' @param to_estimate_u Whether to estimate the units' fixed effects. (default: TRUE)
#' @param to_estimate_v Whether to estimate the time's fixed effects. (default: TRUE)
#' @param to_estimate_b Whether to estimate the unit-time varying covariate parameters. (default: TRUE)
#' @param to_estimate_H Whether to estimate the unit and time covariates linking parameters. (default: TRUE)
#' @param to_add_ID Whether to add an identity matrix to the X and Z matrices. (default: TRUE)
#' @param niter Number of iterations for the coordinate descent steps. (default: 100)
#' @param n_config Number of configurations of lambda that are evaluated in cross-validation (default: 150)
#' @param rel_tol Relative tolerance for the convergence of the coordinate descent steps. (default: 1e-5)
#' @param is_quiet Whether to print the progress of the algorithm. (default: FALSE)
#' @param cv_ratio Share of observations that is used for training in cross-validation. If NA is provided, then the share is equal to the share of non-treated observations in the sample (default: NA)
#' @param num_folds Number of folds in cross-validation (default: 3)
#' @param cube_search Search method for lambda configurations. (default: TRUE)
#' If set to true, then configurations are evaluated in a hypercube fashion with zooming in on best edge every iteration
#' If set to false, lambda values are spaned on a grid.
#' @param post_estimation Whether a post regularization estimation without regularization should be estimated. (default: TRUE)
#' @param impose_null Whether null-hypothesis should be imposed. Implies that all observations also treated observations are used in model estimation. (default: TRUE)
#' @param model_selection_H Whether the unit and time varying covariate link matrix should be regularized. (default: TRUE)
#' @param model_selection_b Whether the unit-time varying covariates parameters should be regularized. (default: TRUE)
#' @param iter_lambda Maximum number of iterations for within the lambda analysis. (default: 100)
#' @param seed Any integer to control the random seed. NA for a random seed (default: NA)
#' @param file_path Path (full or relative) to where the files should be written if write_to_file=true (default: active working directory)
#' @return: Currently no return of the function. Use write_to_file flag to store results to disk.
#' @examples 
#' mcms_lambda_analysis(M, X, Z, B, mask, num_B_cov)

mcms_lambda_analysis=function(M, 
                              mask,
                              X = NA, 
                              Z = NA, 
                              B = NA, 
                              num_B_cov = NA, 
                              to_normalize = TRUE, 
                              to_estimate_u = TRUE, 
                              to_estimate_v = TRUE, 
                              to_estimate_b = TRUE, 
                              to_estimate_H = TRUE, 
                              to_add_ID = FALSE, 
                              niter = 100, 
                              n_config = 100, 
                              rel_tol = 1e-5, 
                              is_quiet = TRUE, 
                              n_lambda = 50, 
                              cv_ratio = NA, 
                              num_folds = 3, 
                              cube_search = TRUE, 
                              post_estimation = TRUE, 
                              impose_null = TRUE, 
                              model_selection_H = TRUE, 
                              model_selection_b = TRUE, 
                              iter_lambda = 100,
                              seed = NA,
                              file_path = './'){
  
  start_time=Sys.time()
  # Convert the path to the full path if necessary
  if(isNA(file_path)){
    write_to_file=FALSE
    file_path_full=""
  } else {
    write_to_file=TRUE
    file_path_full=getAbsolutePath(file_path)
  }
  
  # Check input parameters (additional checks are performed in the cpp code)
  validate_input_parameters(M = M, 
                            mask = mask, 
                            X = X, 
                            Z = Z, 
                            B = B, 
                            num_B_cov = num_B_cov, 
                            to_add_ID = to_add_ID, 
                            to_estimate_u = to_estimate_u, 
                            to_estimate_v = to_estimate_v, 
                            to_estimate_b = to_estimate_b, 
                            to_estimate_H = to_estimate_H, 
                            niter = niter, 
                            rel_tol = rel_tol, 
                            to_normalize = to_normalize, 
                            cv_ratio = cv_ratio,
                            num_folds = num_folds,
                            n_config = n_config, 
                            cube_search = cube_search, 
                            is_quiet = is_quiet, 
                            post_estimation = post_estimation, 
                            impose_null = impose_null,
                            model_selection_H = model_selection_H, 
                            model_selection_b = model_selection_b,
                            seed = seed,
                            n_lambda = n_lambda)
  
  # Set default covariate matrices for seamless estimation in case they are not specified by the user
  if(isNA(X)){X=matrix(data=1, nrow=nrow(M), ncol=1)}
  if(isNA(Z)){Z=matrix(data=1, nrow=1, ncol=ncol(M))}
  if(isNA(B)){B=matrix(data=0, nrow=nrow(M), ncol=ncol(M))}
  
  
  # Set estimation parameters in case they have not been specified by the user
  
  # set a random seed if none is specified
  if(isNA(seed)){
    seed=sample.int(99999,1)
  }
  
  # take share of not-treated observations
  if(isNA(cv_ratio)){
    cv_ratio=mean(mask)
  }
  
  cpp_output=mc_ms_lambda_analysis(M=M, 
                                   X=X, 
                                   Z=t(Z), 
                                   B=B, 
                                   mask=mask, 
                                   num_B_cov=num_B_cov, 
                                   file_path=file_path_full,
                                   to_normalize=to_normalize, 
                                   to_estimate_u=to_estimate_u, 
                                   to_estimate_v=to_estimate_v, 
                                   to_estimate_b=to_estimate_b, 
                                   to_estimate_H=to_estimate_H, 
                                   to_add_ID=to_add_ID, 
                                   niter=niter, 
                                   n_config=n_config, 
                                   rel_tol=rel_tol, 
                                   is_quiet=is_quiet, 
                                   n_lambda=n_lambda, 
                                   cv_ratio=cv_ratio, 
                                   num_folds=num_folds, 
                                   cube_search=cube_search, 
                                   post_estimation=post_estimation, 
                                   impose_null=impose_null,
                                   model_selection_H=model_selection_H, 
                                   model_selection_b=model_selection_b,
                                   seed=seed,
                                   write_to_file=write_to_file,
                                   iter_lambda=iter_lambda)
  
  duration <-  abs(as.numeric(difftime(Sys.time(), start_time, units="secs")))
  if(duration>86400){
    cat(sprintf("Runtime: %02d days, %02d hours,  %02d mins, %02d seconds\n", duration %/% 86400, duration %% 86400 %/% 3600, duration %% 3600 %/% 60, duration %% 60 %/% 1))	
  } else if(duration>3600){
    cat(sprintf("Runtime: %02d hours,  %02d mins, %02d seconds\n", duration %/% 3600, duration %% 3600 %/% 60, duration %% 60 %/% 1))	
  } else if(duration>60){
    cat(sprintf("Runtime: %02d mins, %02d seconds\n", duration %/% 60, duration %% 60 %/% 1))	
  } else {
    cat(sprintf("Runtime: %02d seconds\n", duration %/% 1))	
  } 
  
  return(cpp_output)
}



