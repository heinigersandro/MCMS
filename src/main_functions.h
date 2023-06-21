

#ifndef MAIN_FUNCTIONS_H_
#define MAIN_FUNCTIONS_H_

using namespace Eigen;
using namespace Rcpp;

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
////// Export functions to use in R
///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

// Determine max lambda configuration
List mc_ms_lam_range(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix B, NumericMatrix mask, int num_B_cov, bool to_add_ID=true, bool to_estimate_u=true, bool to_estimate_v=true, int niter=100, double rel_tol=1e-5, bool to_normalize=true, bool is_quiet=false);

// Fit particular lambda configuration
List mc_ms_fit(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix B, NumericMatrix mask, double lambda_L, double lambda_H, double lambda_b, int num_B_cov, bool to_add_ID=true, bool to_estimate_u=true, bool to_estimate_v=true, bool to_estimate_b=true, bool to_estimate_H=true, int niter=100, double rel_tol=1e-5, bool to_normalize=true, bool is_quiet=false, bool post_estimation=true);

// Cross-validation function
List mc_ms_cv(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix B, NumericMatrix mask, int num_B_cov, bool to_add_ID=true, bool to_estimate_u=true, bool to_estimate_v=true, bool to_estimate_b=true, bool to_estimate_H=true, int niter=100, double rel_tol=1e-5, bool to_normalize=true, double cv_ratio = 0.8, int num_folds = 2, int n_config=300, bool cube_search=true, bool is_quiet = true, bool post_estimation = true, bool model_selection_H=true, bool model_selection_b=true, bool return_mse=true, bool return_1se=false, int seed=127127);
int mc_ms_cv_cpp(MatrixXd M, MatrixXd X, MatrixXd Z, MatrixXd B, MatrixXd mask, std::string file_path, int num_B_cov, bool to_add_ID=true, bool to_estimate_u=true, bool to_estimate_v=true, bool to_estimate_b=true, bool to_estimate_H=true, int niter=100, double rel_tol=1e-5, bool to_normalize=true, double cv_ratio = 0.8, int num_folds = 2, int n_config=300, bool cube_search=true, bool is_quiet = true, bool post_estimation = true, bool model_selection_H=true, bool model_selection_b=true, bool return_mse=true, bool return_1se=false, int seed=127127);

// Estimate deviations from optimal lambda
int mc_ms_lambda_analysis(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix B, NumericMatrix mask, String file_path, int num_B_cov, bool to_normalize=true, bool to_estimate_u=true, bool to_estimate_v=true, bool to_estimate_b=true, bool to_estimate_H=true, bool to_add_ID=false, int niter=100, int n_config=100, double rel_tol=1e-5, bool is_quiet = true, int n_lambda = 50, double cv_ratio = 0.8, int num_folds = 2, bool cube_search=true, bool post_estimation = true, bool model_selection_H=true, bool model_selection_b=true, int seed=127127, bool write_to_file=true, int iter_lambda=1000);
int mc_ms_lambda_analysis_cpp(MatrixXd M_, MatrixXd X_, MatrixXd Z_, MatrixXd B_, MatrixXd mask_, std::string file_path, int num_B_cov, bool to_normalize=true, bool to_estimate_u=true, bool to_estimate_v=true, bool to_estimate_b=true, bool to_estimate_H=true, bool to_add_ID=false, int niter=100, int n_config=100, double rel_tol=1e-5, bool is_quiet = true, int n_lambda = 50, double cv_ratio = 0.8, int num_folds = 2, bool cube_search=true, bool post_estimation = true, bool model_selection_H=true, bool model_selection_b=true, int seed=127127, bool write_to_file=true, int iter_lambda=1000);

#endif /* MAIN_FUNCTIONS_H_ */
