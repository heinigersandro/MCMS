
#' This function is used to estimate the maximum lambda_L, lambda_H, and lambda_b values for the MCNNM algorithm.
#'
#' @param M Outcome matrix
#' @param X Units covariate matrix (Time-invariant)
#' @param Z Time covariate matrix (Unit-invariant)
#' @param B Unit-Time covariate matrix
#' @param mask Treatment mask matrix (1 if control, 0 if treated)
#' @param num_B_cov Number of unit-time covariates
#' @param to_add_ID Whether to add an identity matrix to the X and Z matrices. (default: TRUE)
#' @param to_estimate_u Whether to estimate the units' fixed effects. (default: TRUE)
#' @param to_estimate_v Whether to estimate the time's fixed effects. (default: TRUE)
#' @param niter Number of iterations for the coordinate descent steps. (default: 100)
#' @param rel_tol Relative tolerance for the convergence of the coordinate descent steps. (default: 1e-5)
#' @param to_normalize Whether to normalize the X, Z, and B matrices. (default: TRUE)
#' @param is_quiet Whether to print the progress of the algorithm. (default: FALSE)
#' @return (List) List of the maximum lambda_L, lambda_H, and lambda_b values
#' @examples 
#' mcms_lam_range(M, X, Z, B, mask, num_B_cov)

mcms_lam_range=function(M, X, Z, B, mask, num_B_cov, 
						to_add_ID=TRUE, to_estimate_u=TRUE, to_estimate_v=TRUE,
						niter=1e-5,rel_tol=1e-5, to_normalize=TRUE, is_quiet=FALSE){
	start_time=Sys.time()
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

#' This function is used to estimate the MCNNM model with given lambda values for L,H, and b.
#' If post_estimation is set, then the estimated parameters are returned additionally in the list.
#' 
#' @param M Outcome matrix
#' @param X Units covariate matrix (Time-invariant)
#' @param Z Time covariate matrix (Unit-invariant)
#' @param B Unit-Time covariate matrix
#' @param mask Treatment mask matrix (1 if control, 0 if treated)
#' @param lambda_L lambda value for rank-regularization
#' @param lambda_H lambda value for unit and time covariate linking matrix regularization
#' @param lambda_b lambda value for unit-time covariate regularization
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
#' @return: List containing the following elements:
#' 	    - "u": Estimated units' fixed effects
#' 		- "v": Estimated time's fixed effects
#' 		- "B": Estimated unit-time covariates' coefficients
#' 		- "H": Estimated H matrix
#' 		- "tau": The estimate treatment effect
#' 		- "lambda_L": Estimated lambda_L value
#' 		- "lambda_H": Estimated lambda_H value
#' 		- "lambda_b": Estimated lambda_b value
#' @examples 
#' mcms_fit(M, X, Z, B, mask, lambda_L, lambda_H, lambda_b, num_B_cov)

mcms_fit=function(M, 
				  X, 
				  Z, 
				  B, 
				  mask, 
				  lambda_L, 
				  lambda_H, 
				  lambda_b, 
				  num_B_cov, 
				  to_add_ID=TRUE, 
				  to_estimate_u=TRUE, 
				  to_estimate_v=TRUE, 
				  to_estimate_b=TRUE, 
				  to_estimate_H=TRUE, 
				  niter=100, 
				  rel_tol=1e-5, 
				  to_normalize=TRUE, 
				  is_quiet=FALSE, 
				  post_estimation=TRUE){
	start_time=Sys.time()
	
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
						 post_estimation)
	
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

#' This function performs cross-validation to estimate the optimal lambda values and runs the mcnnm_wc model.	
#' If post_estimation is set, then the estimated parameters are returned additionally in the list.
#' 
#' @param M Outcome matrix
#' @param X Units covariate matrix (Time-invariant)
#' @param Z Time covariate matrix (Unit-invariant)
#' @param B Unit-Time covariate matrix
#' @param mask Treatment mask matrix (1 if control, 0 if treated)
#' @param num_B_cov Number of unit-time covariates
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
#' @param model_selection_H Whether the unit and time varying covariate link matrix should be regularized. (default: TRUE)
#' @param model_selection_b Whether the unit-time varying covariates parameters should be regularized. (default: TRUE)
#' @param seed Any integer to control the random seed. NA for a random seed (default: NA)
#' @return: List containing the following elements:
#' 	    - "u": Estimated units' fixed effects
#' 		- "v": Estimated time's fixed effects
#' 		- "B": Estimated unit-time covariates' coefficients
#' 		- "H": Estimated H matrix
#' 		- "tau": The estimate treatment effect
#' 		- "lambda_L": Estimated lambda_L value
#' 		- "lambda_H": Estimated lambda_H value
#' 		- "lambda_b": Estimated lambda_b value
#' @examples 
#' mcms_cv(M, X, Z, B, mask, num_B_cov)
#' 
mcms_cv=function(M, X, Z, B, mask, num_B_cov, 
				 to_add_ID=TRUE, 
				 to_estimate_u=TRUE, 
				 to_estimate_v=TRUE, 
				 to_estimate_b=TRUE, 
				 to_estimate_H=TRUE, 
				 niter=100, 
				 rel_tol=1e-5, 
				 to_normalize=TRUE, 
				 cv_ratio = NA,
				 cv_criterion="mse",
				 num_folds = 3,
				 n_config=150, 
				 cube_search=TRUE, 
				 is_quiet = TRUE, 
				 post_estimation = TRUE, 
				 model_selection_H=TRUE, 
				 model_selection_b=TRUE,
				 seed=NA){
	start_time=Sys.time()
	
	# set a random seed if none is specified
	if(is.na(seed)){
		seed=sample.int(99999,1)
	}
	
	if(is.na(cv_ratio)){
		cv_ratio=mean(mask)
	}
	
	if(cv_criterion %in% c("mse","both")){
		return_mse=T
	} else {return_mse=F}
	if(cv_criterion %in% c("1.se","both")){
		return_1se=T
	} else {return_1se=F}
	
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
						model_selection_H, 
						model_selection_b,
						return_mse=return_mse,
						return_1se=return_1se,
						seed)
	
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
#' This function estimate deviations from optimal lambda
#' If post_estimation is set, then the estimated parameters are returned additionally in the list.
#' 
#' @param M Outcome matrix
#' @param X Units covariate matrix (Time-invariant)
#' @param Z Time covariate matrix (Unit-invariant)
#' @param B Unit-Time covariate matrix
#' @param mask Treatment mask matrix (1 if control, 0 if treated)
#' @param num_B_cov Number of unit-time covariates
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
#' @param model_selection_H Whether the unit and time varying covariate link matrix should be regularized. (default: TRUE)
#' @param model_selection_b Whether the unit-time varying covariates parameters should be regularized. (default: TRUE)
#' @param iter_lambda Maximum number of iterations for within the lambda analysis. (default: 100)
#' @param seed Any integer to control the random seed. NA for a random seed (default: NA)
#' @param file_path Path (full or relative) to where the files should be written if write_to_file=true (default: active working directory)
#' @return: Currently no return of the function. Use write_to_file flag to store results to disk.
#' @examples 
#' mcms_lambda_analysis(M, X, Z, B, mask, num_B_cov)

mcms_lambda_analysis=function(M, 
							  X, 
							  Z, 
							  B, 
							  mask, 
							  num_B_cov, 
							  to_normalize=TRUE, 
							  to_estimate_u=TRUE, 
							  to_estimate_v=TRUE, 
							  to_estimate_b=TRUE, 
							  to_estimate_H=TRUE, 
							  to_add_ID=FALSE, 
							  niter=100, 
							  n_config=100, 
							  rel_tol=1e-5, 
							  is_quiet = TRUE, 
							  n_lambda = 50, 
							  cv_ratio = NA, 
							  num_folds = 3, 
							  cube_search=TRUE, 
							  post_estimation = TRUE, 
							  model_selection_H=TRUE, 
							  model_selection_b=TRUE, 
							  iter_lambda=100,
							  seed=NA,
							  file_path='./'){
	start_time=Sys.time()
	# Convert the path to the full path if necessary
	if(is.na(file_path)){
		write_to_file=FALSE
		file_path_full=""
	} else {
		write_to_file=TRUE
		file_path_full=getAbsolutePath(file_path)
	}
	# set a random seed if none is specified
	if(is.na(seed)){
		seed=sample.int(99999,1)
	}
	
	if(is.na(cv_ratio)){
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



