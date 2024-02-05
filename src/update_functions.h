/*
 * update_functions.h
 *
 *  Created on: 28.10.2022
 *      Author: MgrSHeiniger
 */

#ifndef UPDATE_FUNCTIONS_H_
#define UPDATE_FUNCTIONS_H_

using namespace Eigen;
using namespace Rcpp;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////// Core Functions : All functions that have _H in the very end of their name,
////////                  consider the case where covariates exist.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double round_to(double value, int digit=1);

bool double_zero(double check_value);

VectorXd reduce_vector(VectorXd long_vec, double threshold);

std::tuple<VectorXd, MatrixXd, MatrixXd> MySVD(const MatrixXd &M);

VectorXd logsp(double start, double end, int num_points);

MatrixXd evaluate_Bb(const std::vector<MatrixXd> &B, VectorXd &b, int num_rows, int num_cols);

ArrayXXd Compute_err_Mat(const MatrixXd &M, const MatrixXd &mask, MatrixXd &L, const MatrixXd &X, const MatrixXd &Z, MatrixXd &H, const std::vector<MatrixXd> &B, VectorXd &u, VectorXd &v, VectorXd &b, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H);

double Compute_treat_eff(const MatrixXd &M, const MatrixXd &mask, MatrixXd &L, const MatrixXd &X, const MatrixXd &Z, MatrixXd &H, const std::vector<MatrixXd> &B, VectorXd &u, VectorXd &v, VectorXd &b, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H);

double Compute_MSE(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, MatrixXd &H, const std::vector<MatrixXd> &B, const MatrixXd &mask, MatrixXd &L, VectorXd &u, VectorXd &v, VectorXd &b, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H);

double Compute_objval(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, MatrixXd &H, const std::vector<MatrixXd> &B, const MatrixXd &mask, MatrixXd &L, VectorXd &u, VectorXd &v, VectorXd &b, double sum_sing_vals, double lambda_L, double lambda_H, double lambda_b, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, bool model_selection_H, bool model_selection_b);

MatrixXd SVT(MatrixXd &U, MatrixXd &V, VectorXd& sing_values, ArrayXd& sing_restrict, double sigma);

std::tuple<MatrixXd, VectorXd> update_L(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, MatrixXd &H, const std::vector<MatrixXd> &B, const MatrixXd &mask, MatrixXd L, VectorXd &u, VectorXd &v, VectorXd &b, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, ArrayXd& sing_restrict, double lambda_L);

void update_u(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, MatrixXd &H, const std::vector<MatrixXd> &B, const MatrixXd &mask, MatrixXd &L, VectorXd &u, VectorXd &v, VectorXd &b,int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H);

void update_v(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, MatrixXd &H, const std::vector<MatrixXd> &B, const MatrixXd &mask, MatrixXd &L, VectorXd &u, VectorXd &v, VectorXd &b,int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H);

void update_b(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, MatrixXd &H, const std::vector<MatrixXd> &B, const MatrixXd &mask, MatrixXd &L, VectorXd &u, VectorXd &v, VectorXd &b,int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, int num_B_cov, double lambda_b, std::vector<double> Vtik2_vec, VectorXi &b_zero);

void update_H(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, MatrixXd &H, const std::vector<MatrixXd> &B, MatrixXd &X2Z2sum, const MatrixXd &mask, MatrixXd &L, VectorXd &u, VectorXd &v, VectorXd &b,int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, double lambda_H, MatrixXi &H_zero);

std::tuple<VectorXd, VectorXd, double, double, double, MatrixXd, std::vector<double>> initialize_uv(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, const std::vector<MatrixXd> &B, const MatrixXd &mask, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, int num_B_cov, int niter, double rel_tol);

std::vector<std::tuple<VectorXd, VectorXd, double, double, double, MatrixXd, std::vector<double>, MatrixXd>> create_folds(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, const std::vector<MatrixXd> &B, const MatrixXd &mask, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, int num_B_cov, int niter, double rel_tol, double cv_ratio, int num_folds, int seed);

std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd, double> NNM_fit(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, const std::vector<MatrixXd> &B, MatrixXd H, MatrixXd &X2Z2sum, const MatrixXd &mask, MatrixXd L, VectorXd u, VectorXd v, VectorXd b, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, int num_B_cov, int niter, double rel_tol, double lambda_L, double lambda_H, double lambda_b, std::vector<double> Vtik2_vec, ArrayXd& sing_restrict, bool is_quiet, bool model_selection_H, bool model_selection_b);

std::vector<std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd, double>> NNM_with_uv_init(const MatrixXd &M, MatrixXd &L_init, const MatrixXd &X, MatrixXd &H_init, const MatrixXd &Z, const std::vector<MatrixXd> &B, MatrixXd &X2Z2sum, const MatrixXd &mask, VectorXd &u_init, VectorXd &v_init, VectorXd &b_init, std::vector<std::tuple<double, double, double>> lambda_tuples, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, int num_B_cov, int niter, double rel_tol, std::vector<double> Vtik2_vec, ArrayXd& sing_restrict, bool is_quiet, bool model_selection_H, bool model_selection_b);

std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd, double, double, double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, double, double, double, double, MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd> Find_optimal_lambda(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, const std::vector<MatrixXd> &B, const MatrixXd &mask, const MatrixXd &mask_null, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, int num_B_cov, int niter, double rel_tol, double cv_ratio, int num_folds, bool is_quiet, bool cube_search, int n_config, bool model_selection_H, bool model_selection_b, bool return_1se, int seed, bool is_lambda_analysis=false, std::string file_path_cpp="./", int n_lambda=100);

void eval_post(MatrixXd &L, MatrixXd &H, VectorXd &u, VectorXd &v, VectorXd &b, const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, const std::vector<MatrixXd> &B_vec, const MatrixXd &mask, MatrixXd X2Z2sum, std::vector<double> Vtik2_vec, int num_B_cov, int H_rows, int H_cols, int num_rows, int num_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, int niter, double rel_tol, bool is_quiet);


#endif /* UPDATE_FUNCTIONS_H_ */
