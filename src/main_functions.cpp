
#include <cmath>
#include <vector>
#include <random>
#include <string>

#include <iostream>
#include <fstream>

#include <Rcpp.h>
#include <RcppEigen.h>
#include <stdlib.h>
#include <omp.h>

#include "main_functions.h"
#include "input_checks.h"
#include "update_functions.h"

using namespace Eigen;
using namespace Rcpp;

// [[Rcpp::plugins(openmp)]]

/*
This script contains the main functions for the MCNNM algorithm with integrated model selection.
*/


/**
    * @brief: This function is used to estimate the maximum lambda_L, lambda_H, and lambda_b values for the MCNNM algorithm.
    * @param M: Outcome matrix
	* @param X: Units covariate matrix (Time-invariant)
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param B: Unit-Time covariate matrix
	* @param mask: Treatment mask matrix (1 if control, 0 if treated)
	* @param num_B_cov: Number of unit-time covariates
	* @param to_add_ID: Whether to add an identity matrix to the X and Z matrices. (default: TRUE)
	* @param to_estimate_u: Whether to estimate the units' fixed effects. (default: TRUE)
	* @param to_estimate_v: Whether to estimate the time's fixed effects. (default: TRUE)
	* @param niter: Number of iterations for the coordinate descent steps. (default: 100)
	* @param rel_tol: Relative tolerance for the convergence of the coordinate descent steps. (default: 1e-5)
	* @param to_normalize: Whether to normalize the X, Z, and B matrices. (default: TRUE)
	* @param is_quiet: Whether to print the progress of the algorithm. (default: FALSE)
	* @return: (List) Lis of the maximum lambda_L, lambda_H, and lambda_b values
	*/
// [[Rcpp::export]]
List mc_ms_lam_range(NumericMatrix M,
						NumericMatrix X, 
						NumericMatrix Z, 
						NumericMatrix B, 
						NumericMatrix mask, 
						int num_B_cov, 
						bool to_add_ID, 
						bool to_estimate_u, 
						bool to_estimate_v, 
						int niter, 
						double rel_tol, 
						bool to_normalize,
						bool is_quiet)
{

	if(!is_quiet){
		std::cout << "Prepare the data." << std::endl;
	}
	// convert the input matrices to Eigen matrices
	const MatrixXd M_(as<MatrixXd>(M));
    const MatrixXd mask_(as<MatrixXd>(mask));
	MatrixXd X_(as<MatrixXd>(X));
	MatrixXd Z_(as<MatrixXd>(Z));
	MatrixXd B_(as<MatrixXd>(B));

	// Extract the number of rows and columns of the outcome matrix
	int num_rows = M_.rows();
    int num_cols = M_.cols();

	// Initialize the vectors that will contain the norms of the columns of X, Z, and B
    VectorXd X_col_norms, Z_col_norms, B_col_norms;

	// prepare the data for the algorithm. 
	// This function will also normalize the X, Z, and B matrices if to_normalize is TRUE
	// This function will also add an identity matrix to the X and Z matrices if to_add_ID is TRUE
	// This function will also extract the norms of the columns of X, Z, and B and store them in the col_norms vectors if to_normalize is TRUE
    prepare_data(M_, X_, Z_, B_, mask_, to_add_ID, num_rows, num_cols, X_col_norms, Z_col_norms, B_col_norms, to_normalize, num_B_cov);

	// Convert the X, Z, and B matrices to constant matrices
	const MatrixXd X_fin(X_);
	const MatrixXd Z_fin(Z_.transpose());
	const std::vector<MatrixXd> B_vec = vectorize_B(B_, num_B_cov);

	// Extract the number of rows and columns of the X and Z matrices after adding the identity matrix
	int H_rows=X_fin.cols();
	int H_cols=Z_fin.rows();
	int H_rows_bef = X_fin.cols() - to_add_ID * num_rows;
	int H_cols_bef = Z_fin.rows() - to_add_ID * num_cols;

	// Initialize scalars, vectors, and matrices that will be used in the algorithm. 
	VectorXd u, v;
	double max_lam_L;
	double max_lam_H;
	double max_lam_b;
	MatrixXd X2Z2sum;
	std::vector<double> Vtik2_vec;

	if(!is_quiet){
		std::cout << "Find max lambda values." << std::endl;
	}

	// get the max values of lambda_L, lambda_H, and lambda_b with the initialize_uv function
	std::tie(u,v, max_lam_L, max_lam_H, max_lam_b, X2Z2sum, Vtik2_vec)=initialize_uv(M_, X_fin, Z_fin, B_vec, mask_, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, num_B_cov, niter, rel_tol);

	return List::create(Named("lambda_L_max") = max_lam_L,
	        Named("lambda_H_max") = max_lam_H,
	        Named("lambda_b_max") = max_lam_b);
}


/**
	* @brief: This function is used to estimate the MCNNM model with given lambda values for L,H, and b.
	* @param M: Outcome matrix
	* @param X: Units covariate matrix (Time-invariant)
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param B: Unit-Time covariate matrix
	* @param mask: Treatment mask matrix (1 if control, 0 if treated)
	* @param lambda_L: lambda_L value
	* @param lambda_H: lambda_H value
	* @param lambda_b: lambda_b value
	* @param num_B_cov: Number of unit-time covariates
	* @param to_add_ID: Whether to add an identity matrix to the X and Z matrices. (default: TRUE)
	* @param to_estimate_u: Whether to estimate the units' fixed effects. (default: TRUE)
	* @param to_estimate_v: Whether to estimate the time's fixed effects. (default: TRUE)
	* @param to_estimate_b: Whether to estimate the unit-time covariates' coefficients. (default: TRUE)
	* @param to_estimate_H: Whether to estimate the unit-time covariates linking matrix (default: TRUE)
	* @param niter: Maximum number of iterations for the coordinate descent steps. (default: 100)
	* @param rel_tol: Relative tolerance for the coordinate descent steps. (default: 1e-5)
	* @param to_normalize: Whether to normalize the X, Z, and B matrices. (default: TRUE)
	* @param is_quiet: Whether to print the progress of the algorithm. (default: TRUE)
	* @param post_estimation: Whether to re-estimate the parameters without regularization after the model selection process. (default: TRUE)
	* @param impose_null: Whether null-hypothesis should be imposed. Implies that all observations also treated observations are used in model estimation. (default: TRUE)

	* @return: List containing the following elements:
	*		- "u": Estimated units' fixed effects
	*		- "v": Estimated time's fixed effects
	*		- "B": Estimated unit-time covariates' coefficients
	*		- "H": Estimated H matrix
	*		- "tau": The estimate treatment effect
	*		- "lambda_L": Estimated lambda_L value
	*		- "lambda_H": Estimated lambda_H value
	*		- "lambda_b": Estimated lambda_b value
	* @note: If post_estimation is set, then the estimated parameters are returned additionally in the list.
*/
// [[Rcpp::export]]
List mc_ms_fit(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix B, NumericMatrix mask, double lambda_L, double lambda_H, double lambda_b, int num_B_cov, bool to_add_ID, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, int niter, double rel_tol, bool to_normalize, bool is_quiet, bool post_estimation, bool impose_null, bool model_selection_H, bool model_selection_b) {

	if(!is_quiet){
		std::cout << "Prepare the data." << std::endl;
	}

	// Convert the R matrices to Eigen matrices
	const MatrixXd M_(as<MatrixXd>(M));
	const MatrixXd mask_(as<MatrixXd>(mask));
	MatrixXd X_(as<MatrixXd>(X));
	MatrixXd Z_(as<MatrixXd>(Z));
	MatrixXd B_(as<MatrixXd>(B));

	// Extract the number of rows and columns of the M matrix
	int num_rows = M_.rows();
	int num_cols = M_.cols();
	const MatrixXd mask_null(impose_null ? MatrixXd::Constant(num_rows, num_cols, 1.0) : as<MatrixXd>(mask));

	// intitialize the empty column norms vectors
	VectorXd X_col_norms, Z_col_norms, B_col_norms;

	// prepare the data for the algorithm
	// This function will also normalize the X, Z, and B matrices if to_normalize is TRUE
	// This function will also add an identity matrix to the X and Z matrices if to_add_ID is TRUE
	// This function will also extract the norms of the columns of X, Z, and B and store them in the col_norms vectors if to_normalize is TRUE
    prepare_data(M_, X_, Z_, B_, mask_, to_add_ID, num_rows, num_cols, X_col_norms, Z_col_norms, B_col_norms, to_normalize, num_B_cov);

	// Convert the Eigen matrices to const Eigen matrices
	const MatrixXd X_fin(X_);
	const MatrixXd Z_fin(Z_.transpose());
	const std::vector<MatrixXd> B_vec = vectorize_B(B_, num_B_cov);

	// Extract the number of rows and columns of the X and Z matrices after adding the identity matrix
	int H_rows=X_fin.cols();
	int H_cols=Z_fin.rows();
	int H_rows_bef = X_fin.cols() - to_add_ID * num_rows;
	int H_cols_bef = Z_fin.rows() - to_add_ID * num_cols;

	// Initialize scalars, vectors, and matrices that will be used in the algorithm. 
	VectorXd u, v;
	double max_lam_L, max_lam_H, max_lam_b;
	MatrixXd X2Z2sum;
	std::vector<double> Vtik2_vec;

	if(!is_quiet){
		std::cout << "Initialize starting values for fixed effects." << std::endl;
	}

	// Get the optimal lambda values for L, H, and b
	std::tie(u, v , max_lam_L, max_lam_H, max_lam_b, X2Z2sum, Vtik2_vec)=initialize_uv(M_, X_fin, Z_fin, B_vec, mask_null, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, num_B_cov, niter, rel_tol);

	// initialize the L, H, and b matrices
    MatrixXd L = MatrixXd::Zero(num_rows, num_cols);
	MatrixXd H = MatrixXd::Zero(H_rows, H_cols);
	VectorXd b = VectorXd::Zero(num_B_cov);
	double finished_early;
	ArrayXd sing_restrict = ArrayXd::Ones(std::min(num_rows, num_cols));

	if(!is_quiet){
		std::cout << "Estimate the model." << std::endl;
	}

	// Run the coordinate descent steps algorithm
	std::tie(L, H, u, v, b, finished_early) = NNM_fit(M_, X_fin, Z_fin, B_vec, H, X2Z2sum, mask_null, L, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, num_B_cov, niter, rel_tol, lambda_L, lambda_H, lambda_b, Vtik2_vec, sing_restrict, is_quiet, model_selection_H, model_selection_b);

	if(!is_quiet){
		std::cout << "Terminated at iteration: " << finished_early << std::endl;
	}

	
	// Compute the treatment effect
	double tau=Compute_treat_eff(M_, mask_, L, X_fin, Z_fin, H, B_vec, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);

	// Store the estimated parameters in a list
	List res=List::create(Named("H") = wrap(H),
			Named("L") = wrap(L),
			Named("u") = wrap(u),
			Named("v") = wrap(v),
			Named("b") = wrap(b),
			Named("tau") = tau,
			Named("lambda_L") = lambda_L,
			Named("lambda_H") = lambda_H,
			Named("lambda_b") = lambda_b);

	// Normalize back the estimated parameters of H and b if to_normalize is TRUE
	if (to_normalize && X_.cols() > 0) {
		res["H"]=normalize_back_rows(res["H"], X_col_norms);
	}
	if (to_normalize && Z_.cols() > 0) {
		res["H"]=normalize_back_cols(res["H"], Z_col_norms);
	}
	if (to_normalize && B_.cols() > 0) {
		res["b"]=normalize_back_vector(res["b"], B_col_norms);
	}

	// If post_estimation is set, then return the estimated parameters additionally in the list
	if(post_estimation){
		eval_post(L, H, u, v, b, M_, X_fin, Z_fin, B_vec, mask_null, X2Z2sum, Vtik2_vec, num_B_cov, H_rows, H_cols, num_rows, num_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, niter, rel_tol, is_quiet);
		double tau_post=Compute_treat_eff(M_, mask_, L, X_fin, Z_fin, H, B_vec, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);

		if (to_normalize && X_.cols() > 0) {
			H=normalize_back_rows(H, X_col_norms);
		}
		if (to_normalize && Z_.cols() > 0) {
			H=normalize_back_cols(H, Z_col_norms);
		}
		if (to_normalize && B_.cols() > 0) {
			b=normalize_back_vector(b, B_col_norms);
		}
		res["H_post"]=wrap(H);
		res["L_post"]=wrap(L);
		res["u_post"]=wrap(u);
		res["v_post"]=wrap(v);
		res["b_post"]=wrap(b);
		res["tau_post"]=tau_post;
	}

    return res;
}

/**
	* @brief: This function performs cross-validation to estimate the optimal lambda values and runs the mcnnm_wc model.	
	* @param M: Outcome matrix
	* @param X: Units covariate matrix (Time-invariant)
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param B: Unit-Time covariate matrix
	* @param mask: Treatment mask matrix (1 if control, 0 if treated)
	* @param file_path: full path to where the files should be written if write_to_file=true
	* @param num_B_cov: Number of unit-time covariates
	* @param to_add_ID: Whether to add an ID column to the unit-time covariate matrix (default: false)
	* @param to_estimate_u: Whether to estimate the unit covariate matrix (default: true)
	* @param to_estimate_v: Whether to estimate the time covariate matrix (default: true)
	* @param to_estimate_b: Whether to estimate the unit-time covariate matrix (default: true)
	* @param to_estimate_H: Whether to estimate the unit-time covariates linking matrix (H) (default: true)
	* @param niter: Number of iterations (default: 100)
	* @param rel_tol: Relative tolerance (default: 1e-5)
	* @param to_normalize: Whether to normalize the covariates (default: true)
	* @param cv_ratio: Ratio of the data to be used for cross-validation (default: 0.8)
	* @param num_folds: Number of folds for cross-validation (default: 2)
	* @param n_config: Number of configurations to be tested (default: 300)
	* @param cube_search: Whether to use cube search for cross-validation (default: true)
	* @param is_quiet: Whether to print the progress (default: false)
	* @param post_estimation: Whether to perform post-estimation (default: true)
	* @param impose_null: Whether null-hypothesis should be imposed. Implies that all observations also treated observations are used in model estimation. (default: TRUE)
	* @param model_selection_H: Whether the unit and time varying variables should be regularized or not (default: true)
	* @param model_selection_b: Whether the unit-time varying variables should be regularized or not (default: true)
	* @param return_mse: Whether the estimates for optimality criterion 'mse' are returned (default: true)
	* @param return_1se: Whether the estimates for optimality criterion 'mse+1se' are returned (default: false)
	* @param seed: Integer to control the random seed (default: 127127)
	* @param write_to_file: Whether the results should be written to a file
	* @return: List containing the following elements:
	*		- "u": Estimated units' fixed effects
	*		- "v": Estimated time's fixed effects
	*		- "B": Estimated unit-time covariates' coefficients
	*		- "H": Estimated H matrix
	*		- "tau": The estimate treatment effect
	*		- "lambda_H": Vector of evaluated lambda_H values
	*		- "lambda_b": Vector of evaluated lambda_b values
	*		- "lambda_L": Vector of evaluated lambda_L values
	*		- "MSE": Vector of MSE values for each configuration
	*		- "min_lambda_H": The optimal lambda_H value
	*		- "min_lambda_L": The optimal lambda_L value
	*		- "min_lambda_b": The optimal lambda_b value
	*		- "min_MSE": The optimal MSE value
	* @note: If post_estimation is set, then the estimated parameters are returned additionally in the list.
	* @note: Depending on return_mse and return_1se, the estimated parameters are returned either or for both in the list.
*/
// [[Rcpp::export]]
List mc_ms_cv(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix B, NumericMatrix mask, int num_B_cov, bool to_add_ID, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, int niter, double rel_tol, bool to_normalize, double cv_ratio, int num_folds, int n_config, bool cube_search, bool is_quiet, bool post_estimation, bool impose_null, bool model_selection_H, bool model_selection_b, bool return_mse, bool return_1se, int seed) {

	if(!is_quiet){
		std::cout << " -> Prepare the data." << std::endl << std::endl;
	}
	// Convert the input matrices to Eigen matrices
	const MatrixXd M_(as<MatrixXd>(M));
	const MatrixXd mask_(as<MatrixXd>(mask));
	MatrixXd X_(as<MatrixXd>(X));
	MatrixXd Z_(as<MatrixXd>(Z));
	MatrixXd B_(as<MatrixXd>(B));

	// extract the number of rows and columns
	int num_rows = M_.rows();
	int num_cols = M_.cols();
	const MatrixXd mask_null(impose_null ? MatrixXd::Constant(num_rows, num_cols, 1.0) : as<MatrixXd>(mask));

	// initialize the col norms vectors
	VectorXd X_col_norms, Z_col_norms, B_col_norms;

	// prepare the data for the algorithm
	// This function will also normalize the X, Z, and B matrices if to_normalize is TRUE
	// This function will also add an identity matrix to the X and Z matrices if to_add_ID is TRUE
	// This function will also extract the norms of the columns of X, Z, and B and store them in the col_norms vectors if to_normalize is TRUE
    prepare_data(M_, X_, Z_, B_, mask_, to_add_ID, num_rows, num_cols, X_col_norms, Z_col_norms, B_col_norms, to_normalize, num_B_cov);

	// convert the matrices to const matrices
	const MatrixXd X_fin(X_);
	const MatrixXd Z_fin(Z_.transpose());
	const std::vector<MatrixXd> B_vec = vectorize_B(B_, num_B_cov);

	// extract the number of rows and columns after the preparation
	int H_rows=X_fin.cols();
	int H_cols=Z_fin.rows();
	int H_rows_bef = X_fin.cols() - to_add_ID * num_rows;
	int H_cols_bef = Z_fin.rows() - to_add_ID * num_cols;

	if(!is_quiet){
		std::cout << " -> Run cross-validation to find optimal lambda values." << std::endl;
	}

	// Initialize scalars, vectors, and matrices that will be used in the algorithm. 
	VectorXd u, v, b, u_1se, v_1se, b_1se;
	MatrixXd L, H, L_1se, H_1se;
	double min_lambda_L,min_lambda_H,min_lambda_b;
	double min_lambda_L_1se, min_lambda_H_1se, min_lambda_b_1se;
	std::vector<double> MSE, lambda_Ls_est, lambda_Hs_est, lambda_bs_est;
	double min_MSE, finished_iter, cv_se;
	bool is_lambda_analysis=false; std::string file_path_cpp="./"; int n_lambda=0;

	// find the optimal lambda_L, lambda_H, and lambda_b
	std::tie(L, H, u, v, b, min_lambda_L,min_lambda_H,min_lambda_b, min_MSE, MSE, lambda_Ls_est, lambda_Hs_est, lambda_bs_est, min_lambda_H_1se, min_lambda_L_1se, min_lambda_b_1se, cv_se, L_1se, H_1se, u_1se, v_1se, b_1se)=Find_optimal_lambda(M_, X_fin, Z_fin, B_vec, mask_, mask_null, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, num_B_cov, niter, rel_tol, cv_ratio, num_folds, is_quiet, cube_search, n_config, model_selection_H, model_selection_b, return_1se, seed, is_lambda_analysis, file_path_cpp, n_lambda);
	if(!is_quiet){
		std::cout << " ---> Cross-validation completed." << std::endl;
	}
	MatrixXd X2Z2sum(H_rows, H_cols);

	// generate the XZ vector and the X2Z2 matrix for the full training mask
	for (int i = 0; i < H_rows; i++) {
		for (int j = 0; j < H_cols; j++) {
			ArrayXXd XZ = (X_fin.col(i) * Z_fin.row(j)).array() * mask_null.array();
			X2Z2sum(i,j) = (XZ * XZ).sum();
		}
	}

	// generate Vtik2 vector for full mask
	MatrixXd B_by_i_over_t = MatrixXd::Zero(num_rows, num_cols);
	std::vector<double> Vtik2_vec;
	Vtik2_vec.reserve(num_B_cov);
	for (int i = 0; i < num_B_cov; i++) {
		for (int t = 0; t < num_cols; t++) {
			B_by_i_over_t.col(t) = B_vec[t].col(i);
		}
		Vtik2_vec.push_back((B_by_i_over_t.array() * B_by_i_over_t.array() * mask_null.array()).sum());
	}

	// write optimal lambda configuration to results file
	List res=List::create(Named("best_lambda_L") = min_lambda_L,
		        Named("best_lambda_H") = min_lambda_H,
		        Named("best_lambda_b") = min_lambda_b,
		        Named("min_MSE") = min_MSE,
				Named("MSE_se") = cv_se,
		        Named("MSE") = wrap(MSE),
		        Named("lambda_L") = wrap(lambda_Ls_est),
		        Named("lambda_H") = wrap(lambda_Hs_est),
		        Named("lambda_b") = wrap(lambda_bs_est));

	ArrayXd sing_restrict = ArrayXd::Ones(std::min(num_rows, num_cols));

	if(return_mse){
		if (!is_quiet) {
			std::cout << " -> Estimate model with mse-optimal lambda values" << std::endl;
		}

		// run the coordinate descent algorithm
		std::tie(L, H, u, v, b, finished_iter)= NNM_fit(M_, X_fin, Z_fin, B_vec, H, X2Z2sum, mask_null, L, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, num_B_cov, 2*niter, rel_tol, min_lambda_L,min_lambda_H,min_lambda_b, Vtik2_vec, sing_restrict, is_quiet, model_selection_H, model_selection_b);
		// compute the treatment effect
		double tau=Compute_treat_eff(M_, mask_, L, X_fin, Z_fin, H, B_vec, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);

		// store the estimated parameters in a list
		res["L"] = wrap(L);
		res["u"] = wrap(u);
		res["v"] = wrap(v);
		res["tau"] = tau;

		// normalize the estimated parameters back to the original scale
		MatrixXd H_temp=H;
		if (to_normalize && X_.cols() > 0) {
			H_temp=normalize_back_rows(H_temp, X_col_norms);
		}
		if (to_normalize && Z_.cols() > 0) {
			H_temp=normalize_back_cols(H_temp, Z_col_norms);
		}
		res["H"]=wrap(H_temp);
		if (to_normalize && B_.cols() > 0) {
			res["b"]=wrap(normalize_back_vector(b, B_col_norms));
		} else {
			res["b"]=wrap(b);
		}

		// if post estimation is required, run the algorithm without regularization.
		if(post_estimation){

			if(!is_quiet){
				std::cout << " ---> Run post-regularization estimation." << std::endl << std::endl;
			}

			// run the coordinate descent algorithm without regularization
			eval_post(L, H, u, v, b, M_, X_fin, Z_fin, B_vec, mask_null, X2Z2sum, Vtik2_vec, num_B_cov, H_rows, H_cols, num_rows, num_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, 2*niter, rel_tol, is_quiet);

			// compute the treatment effect
			double tau_post=Compute_treat_eff(M_, mask_, L, X_fin, Z_fin, H, B_vec, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);

			// renormalize the estimated parameters back to the original scale
			MatrixXd H_temp=H;
			if (to_normalize && X_.cols() > 0) {
				H_temp=normalize_back_rows(H_temp, X_col_norms);
			}
			if (to_normalize && Z_.cols() > 0) {
				H_temp=normalize_back_cols(H_temp, Z_col_norms);
			}
			res["H_post"]=wrap(H_temp);
			if (to_normalize && B_.cols() > 0) {
				res["b_post"]=wrap(normalize_back_vector(b, B_col_norms));
			} else {
				res["b_post"]=wrap(b);
			}

			// add the estimated parameters to the return list
			res["L_post"]=wrap(L);
			res["u_post"]=wrap(u);
			res["v_post"]=wrap(v);
			res["tau_post"]=tau_post;
		}
	}
	if(return_1se){

		res["best_lambda_L_1se"] = min_lambda_L_1se;
		res["best_lambda_H_1se"] = min_lambda_H_1se;
		res["best_lambda_b_1se"] = min_lambda_b_1se;

		if (!is_quiet) {
			std::cout << " -> Estimate model with 1se-optimal lambda values" << std::endl;
		}

		// run the coordinate descent algorithm
		std::tie(L, H, u, v, b, finished_iter)= NNM_fit(M_, X_fin, Z_fin, B_vec, H_1se, X2Z2sum, mask_null, L_1se, u_1se, v_1se, b_1se, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, num_B_cov, 2*niter, rel_tol, min_lambda_L_1se, min_lambda_H_1se, min_lambda_b_1se, Vtik2_vec, sing_restrict, is_quiet, model_selection_H, model_selection_b);
		// compute the treatment effect
		double tau=Compute_treat_eff(M_, mask_, L, X_fin, Z_fin, H, B_vec, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);

		// store the estimated parameters in a list
		res["L_1se"] = wrap(L);
		res["u_1se"] = wrap(u);
		res["v_1se"] = wrap(v);
		res["tau_1se"] = tau;

		// normalize the estimated parameters back to the original scale
		MatrixXd H_temp=H;
		if (to_normalize && X_.cols() > 0) {
			H_temp=normalize_back_rows(H_temp, X_col_norms);
		}
		if (to_normalize && Z_.cols() > 0) {
			H_temp=normalize_back_cols(H_temp, Z_col_norms);
		}
		res["H_1se"]=wrap(H_temp);
		if (to_normalize && B_.cols() > 0) {
			res["b_1se"]=wrap(normalize_back_vector(b, B_col_norms));
		} else {
			res["b_1se"]=wrap(b);
		}

		// if post estimation is required, run the algorithm without regularization.
		if(post_estimation){

			if(!is_quiet){
				std::cout << " ---> Run post-regularization estimation." << std::endl;
			}

			// run the coordinate descent algorithm without regularization
			eval_post(L, H, u, v, b, M_, X_fin, Z_fin, B_vec, mask_null, X2Z2sum, Vtik2_vec, num_B_cov, H_rows, H_cols, num_rows, num_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, 2*niter, rel_tol, is_quiet);

			// compute the treatment effect
			double tau_post=Compute_treat_eff(M_, mask_, L, X_fin, Z_fin, H, B_vec, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);

			// renormalize the estimated parameters back to the original scale
			MatrixXd H_temp=H;
			if (to_normalize && X_.cols() > 0) {
				H_temp=normalize_back_rows(H_temp, X_col_norms);
			}
			if (to_normalize && Z_.cols() > 0) {
				H_temp=normalize_back_cols(H_temp, Z_col_norms);
			}
			res["H_1se_post"]=wrap(H_temp);
			if (to_normalize && B_.cols() > 0) {
				res["b_1se_post"]=wrap(normalize_back_vector(b, B_col_norms));
			} else {
				res["b_1se_post"]=wrap(b);
			}

			// add the estimated parameters to the return list
			res["L_1se_post"]=wrap(L);
			res["u_1se_post"]=wrap(u);
			res["v_1se_post"]=wrap(v);
			res["tau_1se_post"]=tau_post;
		}
	}

	if(!is_quiet){
		std::cout << " -> Estimation terminated." << std::endl;
	}

    return res;
}


// [[Rcpp::export]]
int mc_ms_lambda_analysis(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix B, NumericMatrix mask, String file_path, int num_B_cov, bool to_normalize, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, bool to_add_ID, int niter, int n_config, double rel_tol, bool is_quiet, int n_lambda, double cv_ratio, int num_folds, bool cube_search, bool post_estimation, bool impose_null, bool model_selection_H, bool model_selection_b, int seed, bool write_to_file, int iter_lambda) {

	if(!is_quiet){
		std::cout << " -> Prepare the data." << std::endl;
	}

	bool return_1se=true;
	// Convert the input matrices to Eigen matrices
	const MatrixXd M_(as<MatrixXd>(M));
	const MatrixXd mask_(as<MatrixXd>(mask));
	MatrixXd X_(as<MatrixXd>(X));
	MatrixXd Z_(as<MatrixXd>(Z));
	MatrixXd B_(as<MatrixXd>(B));
	std::string file_path_cpp=file_path;

	// extract the number of rows and columns
	int num_rows = M_.rows();
	int num_cols = M_.cols();
	const MatrixXd mask_null(impose_null ? MatrixXd::Constant(num_rows, num_cols, 1.0) : as<MatrixXd>(mask));

	// initialize the col norms vectors
	VectorXd X_col_norms, Z_col_norms, B_col_norms;

	// prepare the data for the algorithm
	// This function will also normalize the X, Z, and B matrices if to_normalize is TRUE
	// This function will also add an identity matrix to the X and Z matrices if to_add_ID is TRUE
	// This function will also extract the norms of the columns of X, Z, and B and store them in the col_norms vectors if to_normalize is TRUE
	prepare_data(M_, X_, Z_, B_, mask_, to_add_ID, num_rows, num_cols, X_col_norms, Z_col_norms, B_col_norms, to_normalize, num_B_cov);

	// convert the matrices to const matrices
	const MatrixXd X_fin(X_);
	const MatrixXd Z_fin(Z_.transpose());
	const std::vector<MatrixXd> B_vec = vectorize_B(B_, num_B_cov);

	// extract the number of rows and columns after the preparation
	int H_rows=X_fin.cols();
	int H_cols=Z_fin.rows();
	int H_rows_bef = X_fin.cols() - to_add_ID * num_rows;
	int H_cols_bef = Z_fin.rows() - to_add_ID * num_cols;

	if(!is_quiet){
		std::cout << " -> Run cross-validation to find optimal lambda values." << std::endl;
	}

	// Initialize scalars, vectors, and matrices that will be used in the algorithm.
	VectorXd u, v, b, u_1se, v_1se, b_1se;
	MatrixXd L, H, L_1se, H_1se;
	double min_lambda_L,min_lambda_H,min_lambda_b;
	double min_lambda_L_1se,min_lambda_H_1se,min_lambda_b_1se;
	std::vector<double> MSE, lambda_Ls_est, lambda_Hs_est, lambda_bs_est;
	double min_MSE, finished_iter, cv_se;

	// find the optimal lambda_L, lambda_H, and lambda_b
	bool is_lambda_analysis=true;
	std::tie(L, H, u, v, b, min_lambda_L,min_lambda_H,min_lambda_b, min_MSE, MSE, lambda_Ls_est, lambda_Hs_est, lambda_bs_est, min_lambda_H_1se, min_lambda_L_1se, min_lambda_b_1se, cv_se, L_1se, H_1se, u_1se, v_1se, b_1se)=Find_optimal_lambda(M_, X_fin, Z_fin, B_vec, mask_, mask_null, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, num_B_cov, niter, rel_tol, cv_ratio, num_folds, is_quiet, cube_search, n_config, model_selection_H, model_selection_b, return_1se, seed, is_lambda_analysis, file_path_cpp, n_lambda);

	std::vector<std::tuple<double, double, double>> configs_to_estimate;
	configs_to_estimate.reserve(n_lambda*3+1);
	configs_to_estimate.push_back(std::tuple<double, double, double>(min_lambda_H, min_lambda_L, min_lambda_b));

	double lam_L_max = *max_element(std::begin(lambda_Ls_est), std::end(lambda_Ls_est));
	double lam_H_max = *max_element(std::begin(lambda_Hs_est), std::end(lambda_Hs_est));
	double lam_b_max = *max_element(std::begin(lambda_bs_est), std::end(lambda_bs_est));

	VectorXd lambda_Hs=logsp(0.0, lam_H_max, n_lambda);
	VectorXd lambda_Ls=logsp(0.0, lam_L_max, n_lambda);
	VectorXd lambda_bs=logsp(0.0, lam_b_max, n_lambda);

	for(int i=0; i<n_lambda; i++){
		configs_to_estimate.push_back(std::tuple<double, double, double>(lambda_Hs(i),min_lambda_L,min_lambda_b));
		configs_to_estimate.push_back(std::tuple<double, double, double>(min_lambda_H,lambda_Ls(i),min_lambda_b));
		configs_to_estimate.push_back(std::tuple<double, double, double>(min_lambda_H,min_lambda_L,lambda_bs(i)));
	}

	if(!is_quiet){
		std::cout << " ---> Fit all lambda configurations on full sample." << std::endl;
	}

	MatrixXd X2Z2sum(H_rows, H_cols);
	// generate the XZ vector and the X2Z2 matrix for the full training mask
	for (int i = 0; i < H_rows; i++) {
		for (int j = 0; j < H_cols; j++) {
			ArrayXXd XZ = (X_fin.col(i) * Z_fin.row(j)).array() * mask_null.array();
			X2Z2sum(i,j) = (XZ * XZ).sum();
		}
	}

	// generate Vtik2 vector for full mask
	MatrixXd B_by_i_over_t = MatrixXd::Zero(num_rows, num_cols);
	std::vector<double> Vtik2_vec;
	Vtik2_vec.reserve(num_B_cov);
	for (int i = 0; i < num_B_cov; i++) {
		// calculate B_by_i_over_t
		for (int t = 0; t < num_cols; t++) {
			B_by_i_over_t.col(t) = B_vec[t].col(i);
		}
		Vtik2_vec.push_back((B_by_i_over_t.array() * B_by_i_over_t.array() * mask_null.array()).sum());
	}

	ArrayXd sing_restrict = ArrayXd::Ones(std::min(num_rows, num_cols));
	std::vector<std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd, double>> lambda_configs =NNM_with_uv_init(M_, L, X_fin, H, Z_fin, B_vec, X2Z2sum, mask_null, u, v, b,
			configs_to_estimate, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, num_B_cov, iter_lambda, rel_tol, Vtik2_vec, sing_restrict, is_quiet, model_selection_H, model_selection_b);
	if(!is_quiet){
		std::cout << " ---> Compute treatment effects and estimate post-regularization model if requested" << std::endl;
	}

	#if defined(_OPENMP)
		#pragma omp parallel for schedule(dynamic)
	#endif
	for(int i=0; i<(n_lambda*3+1); i++){
		double lambda_H_inner, lambda_L_inner, lambda_b_inner;
		VectorXd u_inner, v_inner, b_inner;
		MatrixXd L_inner, H_inner;
		double finished_iter_inner;
		std::tie(lambda_H_inner, lambda_L_inner, lambda_b_inner)=configs_to_estimate[i];
		std::tie(L_inner, H_inner, u_inner, v_inner, b_inner, finished_iter_inner) = lambda_configs[i];

		double tau=Compute_treat_eff(M_, mask_, L_inner, X_fin, Z_fin, H_inner, B_vec, u_inner, v_inner, b_inner, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);

		MatrixXd H_pre;
		VectorXd b_pre;
		if (to_normalize && X_.cols() > 0) {
			H_pre=normalize_back_rows(H_inner, X_col_norms);
		} else {
			H_pre = H_inner;
		}
		if (to_normalize && Z_.cols() > 0) {
			H_pre=normalize_back_cols(H_pre, Z_col_norms);
		}
		if (to_normalize && B_.cols() > 0) {
			b_pre=normalize_back_vector(b_inner, B_col_norms);
		}

		if(write_to_file){
			std::ofstream myfile;
			myfile.open(file_path_cpp+"/lambda_" + std::to_string(i) + "_H_"+std::to_string(round_to(lambda_H_inner,10))+
									"_L_"+std::to_string(round_to(lambda_L_inner,10))+
									"_b_"+std::to_string(round_to(lambda_b_inner,10))+"_pre.csv");
			myfile << "u:" << std::endl << u_inner << std::endl << std::endl << std::endl;
			myfile << "v:" << std::endl << v_inner << std::endl << std::endl << std::endl;
			myfile << "b:" << std::endl << b_pre << std::endl << std::endl << std::endl;
			myfile << "H:" << std::endl << H_pre << std::endl << std::endl << std::endl;
			myfile << "L:" << std::endl << L_inner << std::endl << std::endl << std::endl;
			myfile << "tau:" << std::endl << tau << std::endl << std::endl << std::endl;
			myfile << "min_lambda_L:" << std::endl << min_lambda_L << std::endl << std::endl;
			myfile << "min_lambda_H:" << std::endl << min_lambda_H << std::endl << std::endl;
			myfile << "min_lambda_b:" << std::endl << min_lambda_b << std::endl << std::endl;
			myfile << "min_lambda_L_1se:" << std::endl << min_lambda_L_1se << std::endl << std::endl;
			myfile << "min_lambda_H_1se:" << std::endl << min_lambda_H_1se << std::endl << std::endl;
			myfile << "min_lambda_b_1se:" << std::endl << min_lambda_b_1se << std::endl << std::endl;
			myfile << "lambda_L_inner:" << std::endl << lambda_L_inner << std::endl << std::endl;
			myfile << "lambda_H_inner:" << std::endl << lambda_H_inner << std::endl << std::endl;
			myfile << "lambda_b_inner:" << std::endl << lambda_b_inner << std::endl << std::endl;
			myfile.close();
		}

		if(post_estimation){
			eval_post(L_inner, H_inner, u_inner, v_inner, b_inner, M_, X_fin, Z_fin, B_vec, mask_null, X2Z2sum, Vtik2_vec, num_B_cov, H_rows, H_cols, num_rows, num_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, iter_lambda, rel_tol, is_quiet);
			double tau_post=Compute_treat_eff(M_, mask_, L_inner, X_fin, Z_fin, H_inner, B_vec, u_inner, v_inner, b_inner, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);
			if (to_normalize && X_fin.cols() > 0) {
				H_inner=normalize_back_rows(H_inner, X_col_norms);
			}
			if (to_normalize && Z_fin.cols() > 0) {
				H_inner=normalize_back_cols(H_inner, Z_col_norms);
			}
			if (to_normalize && B_.cols() > 0) {
				b_inner=normalize_back_vector(b_inner, B_col_norms);
			}
			if(write_to_file) {
				std::ofstream myfile2;
				myfile2.open(file_path_cpp+"/lambda_" + std::to_string(i) + "_H_"+std::to_string(round_to(lambda_H_inner,10))+
												"_L_"+std::to_string(round_to(lambda_L_inner,10))+
												"_b_"+std::to_string(round_to(lambda_b_inner,10))+"_post.csv");
				myfile2 << "u_post:" << std::endl << u_inner << std::endl << std::endl << std::endl;
				myfile2 << "v_post:" << std::endl << v_inner << std::endl << std::endl << std::endl;
				myfile2 << "b_post:" << std::endl << b_inner << std::endl << std::endl << std::endl;
				myfile2 << "H_post:" << std::endl << H_inner << std::endl << std::endl << std::endl;
				myfile2 << "L_post:" << std::endl << L_inner << std::endl << std::endl << std::endl;
				myfile2 << "tau_post:" << std::endl << tau_post << std::endl << std::endl;
				myfile2 << "min_lambda_L:" << std::endl << min_lambda_L << std::endl << std::endl;
				myfile2 << "min_lambda_H:" << std::endl << min_lambda_H << std::endl << std::endl;
				myfile2 << "min_lambda_b:" << std::endl << min_lambda_b << std::endl << std::endl;
				myfile2 << "min_lambda_L_1se:" << std::endl << min_lambda_L_1se << std::endl << std::endl;
				myfile2 << "min_lambda_H_1se:" << std::endl << min_lambda_H_1se << std::endl << std::endl;
				myfile2 << "min_lambda_b_1se:" << std::endl << min_lambda_b_1se << std::endl << std::endl;
				myfile2 << "lambda_L_inner:" << std::endl << lambda_L_inner << std::endl << std::endl;
				myfile2 << "lambda_H_inner:" << std::endl << lambda_H_inner << std::endl << std::endl;
				myfile2 << "lambda_b_inner:" << std::endl << lambda_b_inner << std::endl << std::endl;
				myfile2.close();
			}
		}
	}
	return 0;
}

