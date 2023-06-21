/*
 * input_checks.h
 *
 *  Created on: 28.10.2022
 *      Author: MgrSHeiniger
 */

#ifndef INPUT_CHECKS_H_
#define INPUT_CHECKS_H_

using namespace Eigen;
using namespace Rcpp;

bool mask_check(const MatrixXd &mask);

bool X_size_check(const MatrixXd &M, MatrixXd &X);

bool Z_size_check(const MatrixXd &M, MatrixXd &Z);

bool mask_size_check(const MatrixXd &M, const MatrixXd &mask);

VectorXd normalize(MatrixXd &mat);

VectorXd normalize_B(MatrixXd &mat, int num_B_cov, int num_rows, int num_cols);

MatrixXd normalize_back_rows(MatrixXd H, VectorXd &row_H_scales);

MatrixXd normalize_back_cols(MatrixXd H, VectorXd &col_H_scales);

VectorXd normalize_back_vector(VectorXd vec, VectorXd &vec_scales);

const std::vector<MatrixXd> vectorize_B(MatrixXd &B, int num_B_cov);

bool mcnnm_matrix_check(const MatrixXd &M, MatrixXd &X, MatrixXd &Z, MatrixXd &B, const MatrixXd &mask, bool to_add_ID);

bool mcnnm_tol_check(double rel_tol);

bool mcnnm_cv_check(int n_config, double cv_ratio, int num_folds);

MatrixXd X_add_id(MatrixXd X, int num_rows);

MatrixXd Z_add_id(MatrixXd Z, int num_cols);

void prepare_data(const MatrixXd &M, MatrixXd &X, MatrixXd &Z, MatrixXd &B, const MatrixXd &mask, int to_add_ID, int num_rows, int num_cols, VectorXd &X_col_norms, VectorXd &Z_col_norms, VectorXd &B_col_norms, bool to_normalize, int num_B_cov);

#endif /* INPUT_CHECKS_H_ */
