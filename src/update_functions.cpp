

#include <cmath>
#include <random>
#include <vector>
#include <tuple>

#include <Rcpp.h>
#include <RcppEigen.h>
#include <stdlib.h>
#include <omp.h>

#include <iostream>
#include <fstream>

#include "update_functions.h"

// [[Rcpp::plugins(openmp)]]

// tolerance for checking if a double is zero
double zero_tol=1e-10;
// number of configurations evaluated to search for mse+1se condition
int num_1se_config=48;

using namespace Eigen;
using namespace Rcpp;

/** 
	* @brief: This function rounds a double to a specified number of digits
	* @param value: value to be rounded
	* @param digit: number of digits to round to
	* @return: (double) rounded value
*/
double round_to(double value, int digit){
	// round values to a specified number of digits
    return std::round(value * pow(10,digit)) / pow(10,digit);
}

/**
	* @brief: This function checks if a double is zero. It uses the global variable zero_tol.
	* @param check_value: value to be checked
	* @return: (boolean) true if the value is zero, false otherwise
	* @note: This function is used to avoid numerical errors.
*/
bool double_zero(double check_value){
	return abs(check_value)<zero_tol;
}
/**
	* @brief: This function computes the Singular Value Decomposition and it passes U,V,Sigma.
	* @param M: matrix to be decomposed
	* @return: (tuple) tuple containing Sigma, U and V
*/
std::tuple<VectorXd, MatrixXd, MatrixXd> MySVD(const MatrixXd &M) {

    // This function computes the Singular Value Decomposition and it passes U,V,Sigma.

    JacobiSVD<MatrixXd> svd(M.rows(), M.cols(), ComputeThinV | ComputeThinU);
    svd.compute(M);
    return std::tuple<VectorXd, MatrixXd, MatrixXd>(svd.singularValues(), svd.matrixU(), svd.matrixV());
}
/**
	* @brief: This function creates logarithmically spaced numbers. It starts with start, ends with end, and creates num_points numbers.
	* @param start: start value
	* @param end: end value
	* @param num_points: number of points
	* @return: (VectorXd) containing the logarithmically spaced numbers
	* @note: If num_points is 1, the function returns a vector with the value start.
	* 		If start is below zero, it is set to zero.
*/
VectorXd logsp(double start, double end, int num_points) {

	VectorXd res(num_points);
	start=std::max(start,0.0);
	res[0] = start;
    if (num_points > 1) {
        if(double_zero(start)){
        	// set to a small value because log(0) is infinite
        	start=end/(num_points*num_points);
        }
        // equally distribute in logs
        double log_start=log(start);
        double step_size=(log(end)-log_start) / (num_points-1);

        // take exponential of equally distributed values
        for (int i = 1; i < num_points; i++) {
            res[i] = exp(log_start + i * step_size);
        }
    }
    return res;
}

/**
	* @brief: function evaluate the unit-time varying covariates. It calculates [Bb]_{:,t}=B_t * b
	* @param B: vector of matrices B_t
	* @param b: vector of coefficients b
	* @param num_rows: number of rows of the output matrix (units)
	* @param num_cols: number of columns of the output matrix (time-periods)
	* @return: (MatrixXd) matrix [Bb]_{:,t}=B_t * b
*/
MatrixXd evaluate_Bb(const std::vector<MatrixXd> &B, VectorXd &b, int num_rows, int num_cols) {

	// initilize output matrix
	MatrixXd Bb(num_rows, num_cols);

	// iterate over all time-periods
    for (int i = 0; i < num_cols; ++i) {
        Bb.col(i) = B[i] * b;
    }
    return(Bb);
}

/**
	* @brief: This function computes the projection of M - (L + X*H*Z^T + u1^T + 1v^T) to observed control observations.
	* @param M: Outcome matrix
	* @param mask: Treatment mask matrix for the current fold (1 if control and in training set, 0 otherwise)
	* @param L: rank-regularized matrix of unobserved factors
	* @param X: Units covariate matrix (Time-invariant)
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param H: Linking matrix for unit and time covariates
	* @param B: Vector of Unit-Time covariate matrices
	* @param u: Unit fixed effects
	* @param v: Time fixed effects
	* @param b: Unit-Time covariate coefficients
	* @param num_rows: number of rows of the outcome matrix
	* @param num_cols: number of columns of the outcome matrix
	* @param H_rows: actual number of rows of the linking matrix after adding the identity matrix
	* @param H_cols: actual number of columns of the linking matrix after adding the identity matrix
	* @param H_rows_bef: number of rows of the linking matrix before adding the identity matrix
	* @param H_cols_bef: number of columns of the linking matrix before adding the identity matrix
	* @param to_estimate_u: boolean whether the unit fixed effects are estimated
	* @param to_estimate_v: boolean whether the time fixed effects are estimated
	* @param to_estimate_b: boolean whether the unit-time covariate coefficients are estimated
	* @param to_estimate_H: boolean whether the linking matrix is estimated
	* @return: (ArrayXXd) matrix containing the projection of M - (L + X*H*Z^T + u1^T + 1v^T) to the observed values
*/
ArrayXXd Compute_err_Mat(const MatrixXd &M, const MatrixXd &mask, MatrixXd &L, const MatrixXd &X, const MatrixXd &Z, MatrixXd &H, const std::vector<MatrixXd> &B, VectorXd &u, VectorXd &v, VectorXd &b, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H) {

	// Initialize result matrix by the difference between the outcome and the unobserved factors.
    MatrixXd res(M-L);

	// If the linking matrix is estimated, subtract X * H * Z^T of the result matrix.
    if(to_estimate_H){
    	res -=X.topLeftCorner(num_rows, H_rows_bef) * H.topLeftCorner(H_rows_bef, H_cols_bef) * Z.topLeftCorner(H_cols_bef, num_cols);
    	if (H_cols > H_cols_bef) {
			res -= H.bottomLeftCorner(num_rows, H_cols_bef) * Z.topLeftCorner(H_cols_bef, num_cols);
		}
    	if (H_rows > H_rows_bef) {
			res -= X.topLeftCorner(num_rows, H_rows_bef) * H.topRightCorner(H_rows_bef, num_cols);
		}
    }

	// If the unit fixed effects are estimated, subtract u1^T of the result matrix.
    if(to_estimate_u){
		res -= u * VectorXd::Constant(num_cols, 1).transpose();
	}

	// If the time fixed effects are estimated, subtract 1v^T of the result matrix.
    if(to_estimate_v){
		res -= VectorXd::Constant(num_rows, 1) * v.transpose();
	}

	// If the unit-time covariate coefficients are estimated, subtract B * b of the result matrix.
    if(to_estimate_b){
		res -= evaluate_Bb(B, b, num_rows, num_cols);
	}

	// Return the projection of the result matrix to the observed untreated values.
    return res.array()*mask.array();
}

/** 
	* @brief: This function calculates the treatment effect for the treated.
	* @param M: Outcome matrix
	* @param mask: Treatment mask matrix for the current fold (1 if control and in training set, 0 otherwise)
	* @param L: rank-regularized matrix of unobserved factors
	* @param X: Units covariate matrix (Time-invariant)
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param H: Linking matrix for unit and time covariates
	* @param B: Vector of Unit-Time covariate matrices
	* @param u: Unit fixed effects
	* @param v: Time fixed effects
	* @param b: Unit-Time covariate coefficients
	* @param num_rows: number of rows of the outcome matrix
	* @param num_cols: number of columns of the outcome matrix
	* @param H_rows: actual number of rows of the linking matrix after adding the identity matrix
	* @param H_cols: actual number of columns of the linking matrix after adding the identity matrix
	* @param H_rows_bef: number of rows of the linking matrix before adding the identity matrix
	* @param H_cols_bef: number of columns of the linking matrix before adding the identity matrix
	* @param to_estimate_u: boolean whether the unit fixed effects are estimated
	* @param to_estimate_v: boolean whether the time fixed effects are estimated
	* @param to_estimate_b: boolean whether the unit-time covariate coefficients are estimated
	* @param to_estimate_H: boolean whether the linking matrix is estimated
	* @return: (double) treatment effect of the treated
	* @note: This function should be only used if the training set is the full set of observations 
	*        because it uses the complement of the mask matrix (1-mask) to calculate the treatment effect.

*/
double Compute_treat_eff(const MatrixXd &M, const MatrixXd &mask, MatrixXd &L, const MatrixXd &X, const MatrixXd &Z, MatrixXd &H, const std::vector<MatrixXd> &B, VectorXd &u, VectorXd &v, VectorXd &b, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H) {

	// Initialize result matrix by the difference between the outcome and the unobserved factors.
    MatrixXd res(M-L);

	// If the linking matrix is estimated, subtract X * H * Z^T of the result matrix.
    if(to_estimate_H){
    	res -=X.topLeftCorner(num_rows, H_rows_bef) * H.topLeftCorner(H_rows_bef, H_cols_bef) * Z.topLeftCorner(H_cols_bef, num_cols);
    	if (H_cols > H_cols_bef) {
			res -= H.bottomLeftCorner(num_rows, H_cols_bef) * Z.topLeftCorner(H_cols_bef, num_cols);
		}
    	if (H_rows > H_rows_bef) {
			res -= X.topLeftCorner(num_rows, H_rows_bef) * H.topRightCorner(H_rows_bef, num_cols);
		}
    }

	// If the unit fixed effects are estimated, subtract u1^T of the result matrix.
    if(to_estimate_u){
		res -= u * VectorXd::Constant(num_cols, 1).transpose();
	}

	// If the time fixed effects are estimated, subtract 1v^T of the result matrix.
    if(to_estimate_v){
		res -= VectorXd::Constant(num_rows, 1) * v.transpose();
	}

	// If the unit-time covariate coefficients are estimated, subtract B * b of the result matrix.
    if(to_estimate_b){
		res -= evaluate_Bb(B, b, num_rows, num_cols);
	}

	// Number of treated observations
    double n_treat=(num_rows*num_cols)-mask.array().sum();

	// average difference between for the treated observations give the ATET
    return (res.array()*(ArrayXXd::Constant(num_rows, num_cols, 1.0) - mask.array())).sum() / n_treat;
}

/**
	* @brief This function calculates the mean squared error.
	* @param M: Outcome matrix
	* @param mask: Treatment mask matrix for the current fold (1 if control and in training set, 0 otherwise)
	* @param L: rank-regularized matrix of unobserved factors
	* @param X: Units covariate matrix (Time-invariant)
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param H: Linking matrix for unit and time covariates
	* @param B: Vector of Unit-Time covariate matrices
	* @param u: Unit fixed effects
	* @param v: Time fixed effects
	* @param b: Unit-Time covariate coefficients
	* @param num_rows: number of rows of the outcome matrix
	* @param num_cols: number of columns of the outcome matrix
	* @param H_rows: actual number of rows of the linking matrix after adding the identity matrix
	* @param H_cols: actual number of columns of the linking matrix after adding the identity matrix
	* @param H_rows_bef: number of rows of the linking matrix before adding the identity matrix
	* @param H_cols_bef: number of columns of the linking matrix before adding the identity matrix
	* @param to_estimate_u: boolean whether the unit fixed effects are estimated
	* @param to_estimate_v: boolean whether the time fixed effects are estimated
	* @param to_estimate_b: boolean whether the unit-time covariate coefficients are estimated
	* @param to_estimate_H: boolean whether the linking matrix is estimated
	* @return: (double) mean squared error
*/
double Compute_MSE(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, MatrixXd &H, const std::vector<MatrixXd> &B, const MatrixXd &mask, MatrixXd &L, VectorXd &u, VectorXd &v, VectorXd &b, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H) {

	// Number of observed control observations in the training set
    double valid_size = mask.sum();

	// Calculate the error matrix
    ArrayXXd err_mask = Compute_err_Mat(M, mask, L, X, Z, H, B, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);

    // Calculate the mean squared error
	double res = (err_mask * err_mask).sum() / valid_size;
    return res;
}

/**
	* @brief This function computes objective function which is the MSE plus nuclear norm of L and also element-wise l1 norm of H and b.
	* @param M: Outcome matrix
	* @param mask: Treatment mask matrix for the current fold (1 if control and in training set, 0 otherwise)
	* @param L: rank-regularized matrix of unobserved factors
	* @param X: Units covariate matrix (Time-invariant)
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param H: Linking matrix for unit and time covariates
	* @param B: Vector of Unit-Time covariate matrices
	* @param u: Unit fixed effects
	* @param v: Time fixed effects
	* @param b: Unit-Time covariate coefficients
	* @param num_rows: number of rows of the outcome matrix
	* @param num_cols: number of columns of the outcome matrix
	* @param H_rows: actual number of rows of the linking matrix after adding the identity matrix
	* @param H_cols: actual number of columns of the linking matrix after adding the identity matrix
	* @param H_rows_bef: number of rows of the linking matrix before adding the identity matrix
	* @param H_cols_bef: number of columns of the linking matrix before adding the identity matrix
	* @param to_estimate_u: boolean whether the unit fixed effects are estimated
	* @param to_estimate_v: boolean whether the time fixed effects are estimated
	* @param to_estimate_b: boolean whether the unit-time covariate coefficients are estimated
	* @param to_estimate_H: boolean whether the linking matrix is estimated
	* @param lambda_L: regularization parameter for the nuclear norm of L
	* @param lambda_H: regularization parameter for the element-wise l1 norm of H
	* @param lambda_b: regularization parameter for the element-wise l1 norm of b
	* @return: (double) MSE plus nuclear norm of L and also element-wise l1 norm of H and b
*/
double Compute_objval(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, MatrixXd &H, const std::vector<MatrixXd> &B, const MatrixXd &mask, MatrixXd &L, VectorXd &u, VectorXd &v, VectorXd &b, double sum_sing_vals, double lambda_L, double lambda_H, double lambda_b, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, bool model_selection_H, bool model_selection_b) {
	// Calculate the mean squared error
    double obj_val = Compute_MSE(M, X, Z, H, B, mask, L, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);

	// add nuclear norm of L
    obj_val+=lambda_L * sum_sing_vals;

	// account for the l1 norm of H and b
	if(model_selection_H){
		obj_val+=lambda_H*(H.array().abs().sum());
	}
    if(model_selection_b){
    	obj_val+=lambda_b*(b.array().abs().sum());
    }

    return obj_val;
}

/**
 * @brief: Given a singular value decomposition and a threshold sigma, this function applies the Singular Value Thresholding operator.
 * Furthermore, it updates the singular values with the truncated version (new singular values of L) which would
 * then be used to compute objective value.
 * @param U: left singular vectors
 * @param V: right singular vectors
 * @param sing_values: singular values
 * @param sing_restrict: vector of 0s and 1s which restricts the singular values to be zero or not
 * @param sigma: threshold
 * @return: (MatrixXd) Updated L
*/
MatrixXd SVT(MatrixXd &U, MatrixXd &V, VectorXd& sing_values, ArrayXd& sing_restrict, double sigma) {

    // truncate the singular values based on threshold
	VectorXd trunc_sing = sing_values - VectorXd::Constant(sing_values.size(), sigma);
    trunc_sing = trunc_sing.cwiseMax(0);
    // in post estimation, truncate the number of non-zero singular values
    sing_values = trunc_sing.array()*sing_restrict;
    return U * sing_values.asDiagonal() * V.transpose();
}

/**
	* @brief: This function updates L in coordinate descent algorithm. The core step of this part is
    * performing a SVT update. Furthermore, it saves the singular values (needed to compute objective value) later.
    * This would help us to only perform one SVD per iteration.
	* @param M: Outcome matrix
	* @param mask: Treatment mask matrix for the current fold (1 if control and in training set, 0 otherwise)
	* @param L: rank-regularized matrix of unobserved factors
	* @param X: Units covariate matrix (Time-invariant)
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param H: Linking matrix for unit and time covariates
	* @param B: Vector of Unit-Time covariate matrices
	* @param u: Unit fixed effects
	* @param v: Time fixed effects
	* @param b: Unit-Time covariate coefficients
	* @param num_rows: number of rows of the outcome matrix
	* @param num_cols: number of columns of the outcome matrix
	* @param H_rows: actual number of rows of the linking matrix after adding the identity matrix
	* @param H_cols: actual number of columns of the linking matrix after adding the identity matrix
	* @param H_rows_bef: number of rows of the linking matrix before adding the identity matrix
	* @param H_cols_bef: number of columns of the linking matrix before adding the identity matrix
	* @param to_estimate_u: boolean whether the unit fixed effects are estimated
	* @param to_estimate_v: boolean whether the time fixed effects are estimated
	* @param to_estimate_b: boolean whether the unit-time covariate coefficients are estimated
	* @param to_estimate_H: boolean whether the linking matrix is estimated
	* @param sing_restrict: vector of singular values of L
	* @param lambda_L: regularization parameter for the nuclear norm of L
	* @return: (tuple) Updated L and the singular values of L 
	* @note: updating is NOT in-place
*/
std::tuple<MatrixXd, VectorXd> update_L(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, MatrixXd &H, const std::vector<MatrixXd> &B, const MatrixXd &mask, MatrixXd L, VectorXd &u, VectorXd &v, VectorXd &b, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, ArrayXd& sing_restrict, double lambda_L) {

	// Compute the projection matrix of L and the error matrix
    MatrixXd proj = L.array()+Compute_err_Mat(M, mask, L, X, Z, H, B, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);
    
	// initialize empty matrices for the singular value decomposition
	MatrixXd U;
    MatrixXd V;
    VectorXd sing;

	// perform the singular value decomposition
    std::tie(sing, U, V) = MySVD(proj);

	// update the L matrix using lambda_L*train_size/2 as the threshold
	MatrixXd L_upd = SVT(U, V, sing, sing_restrict, lambda_L);
    return std::tuple<MatrixXd, VectorXd>(L_upd, sing);
}

/**
	* @brief: This function updates u in coordinate descent algorithm, when covariates are available.
	* @param M: Outcome matrix
	* @param mask: Treatment mask matrix for the current fold (1 if control and in training set, 0 otherwise)
	* @param L: rank-regularized matrix of unobserved factors
	* @param X: Units covariate matrix (Time-invariant)
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param H: Linking matrix for unit and time covariates
	* @param B: Vector of Unit-Time covariate matrices
	* @param u: Unit fixed effects
	* @param v: Time fixed effects
	* @param b: Unit-Time covariate coefficients
	* @param num_rows: number of rows of the outcome matrix
	* @param num_cols: number of columns of the outcome matrix
	* @param H_rows: actual number of rows of the linking matrix after adding the identity matrix
	* @param H_cols: actual number of columns of the linking matrix after adding the identity matrix
	* @param H_rows_bef: number of rows of the linking matrix before adding the identity matrix
	* @param H_cols_bef: number of columns of the linking matrix before adding the identity matrix
	* @param to_estimate_u: boolean whether the unit fixed effects are estimated
	* @param to_estimate_v: boolean whether the time fixed effects are estimated
	* @param to_estimate_b: boolean whether the unit-time covariate coefficients are estimated
	* @param to_estimate_H: boolean whether the linking matrix is estimated
	* @return: void
	* @note: updating is in-place
*/
void update_u(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, MatrixXd &H, const std::vector<MatrixXd> &B, const MatrixXd &mask, MatrixXd &L, VectorXd &u, VectorXd &v, VectorXd &b, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H) {

	// set to_estimate_u to false, so that the error matrix is computed without the unit fixed effects
	to_estimate_u = false;

	// compute the error matrix
	MatrixXd j_mask = Compute_err_Mat(M, mask, L, X, Z, H, B, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);
	// compute the sum of the error matrix for each row
	VectorXd j_mask_sum=j_mask.rowwise().sum();

	// divide the sum of the error matrix for each row by the number of observed observations in the training set
    for (int i = 0; i < num_rows; i++) {
    	double l=(mask.row(i).array()>0).count();
        if (l > 0) {
			// update the unit fixed effects
            u(i) = j_mask_sum(i) / l;
        }
    }
}

/**
	* @brief: This function updates v in coordinate descent algorithm, when covariates are available.
	* @param M: Outcome matrix
	* @param mask: Treatment mask matrix for the current fold (1 if control and in training set, 0 otherwise)
	* @param L: rank-regularized matrix of unobserved factors
	* @param X: Units covariate matrix (Time-invariant)
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param H: Linking matrix for unit and time covariates
	* @param B: Vector of Unit-Time covariate matrices
	* @param u: Unit fixed effects
	* @param v: Time fixed effects
	* @param b: Unit-Time covariate coefficients
	* @param num_rows: number of rows of the outcome matrix
	* @param num_cols: number of columns of the outcome matrix
	* @param H_rows: actual number of rows of the linking matrix after adding the identity matrix
	* @param H_cols: actual number of columns of the linking matrix after adding the identity matrix
	* @param H_rows_bef: number of rows of the linking matrix before adding the identity matrix
	* @param H_cols_bef: number of columns of the linking matrix before adding the identity matrix
	* @param to_estimate_u: boolean whether the unit fixed effects are estimated
	* @param to_estimate_v: boolean whether the time fixed effects are estimated
	* @param to_estimate_b: boolean whether the unit-time covariate coefficients are estimated
	* @param to_estimate_H: boolean whether the linking matrix is estimated
	* @return: void
	* @note: updating is in-place
*/
void update_v(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, MatrixXd &H, const std::vector<MatrixXd> &B, const MatrixXd &mask, MatrixXd &L, VectorXd &u, VectorXd &v, VectorXd &b, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H) {

	// set to_estimate_v to false, so that the error matrix is computed without the time fixed effects
	to_estimate_v = false;

	// compute the error matrix
	MatrixXd j_mask = Compute_err_Mat(M, mask, L, X, Z, H, B, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);
	// compute the sum of the error matrix for each column
	VectorXd j_mask_sum=j_mask.colwise().sum();

	// divide the sum of the error matrix for each column by the number of observed observations in the training set
    for (int i = 1; i < num_cols; i++) {
    	double l=(mask.col(i).array()>0).count();
        if (l > 0) {
			// update the time fixed effects
            v(i) = j_mask_sum(i) / l;
        }
    }
}

/** 
	* @brief: This function updates b in coordinate descent algorithm, when covariates are available. It uses lambda_b as the regularization parameter.
	* @param M: Outcome matrix
	* @param mask: Treatment mask matrix for the current fold (1 if control and in training set, 0 otherwise)
	* @param L: rank-regularized matrix of unobserved factors
	* @param X: Units covariate matrix (Time-invariant)
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param H: Linking matrix for unit and time covariates
	* @param B: Vector of Unit-Time covariate matrices
	* @param u: Unit fixed effects
	* @param v: Time fixed effects
	* @param b: Unit-Time covariate coefficients
	* @param num_rows: number of rows of the outcome matrix
	* @param num_cols: number of columns of the outcome matrix
	* @param H_rows: actual number of rows of the linking matrix after adding the identity matrix
	* @param H_cols: actual number of columns of the linking matrix after adding the identity matrix
	* @param H_rows_bef: number of rows of the linking matrix before adding the identity matrix
	* @param H_cols_bef: number of columns of the linking matrix before adding the identity matrix
	* @param to_estimate_u: boolean whether the unit fixed effects are estimated
	* @param to_estimate_v: boolean whether the time fixed effects are estimated
	* @param to_estimate_b: boolean whether the unit-time covariate coefficients are estimated
	* @param to_estimate_H: boolean whether the linking matrix is estimated
	* @param num_B_cov: number of unit-time covariates
	* @param lambda_b: regularization parameter for unit-time covariates
	* @param Vtik2_vec: vector of Vtik2 values for each time-period
	* @return: void
	* @note: updating is in-place
*/
void update_b(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, MatrixXd &H, const std::vector<MatrixXd> &B, const MatrixXd &mask, MatrixXd &L, VectorXd &u, VectorXd &v, VectorXd &b,int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, int num_B_cov, double lambda_b, std::vector<double> Vtik2_vec, VectorXi &b_zero) {

    // structure is retrieved from here: https://xavierbourretsicotte.github.io/lasso_implementation.html
    ArrayXXd B_by_i_over_t = ArrayXXd::Zero(num_rows, num_cols);

	// compute the error matrix. Here it is not possible to set to_estimate_b to false because we are evaluating one time-period at the time and have to take into account all other time-periods
	ArrayXXd j_mask = Compute_err_Mat(M, mask, L, X, Z, H, B, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);

	// loop over all b coefficients
    for (int i = 0; i < num_B_cov; i++) {
		//Do only run positions that have not been thresholded out
    	if(b_zero(i)!=0){
    		// Get the i-th (current B coef) column of all time-periods multiplied by the mask
			for (int t = 0; t < num_cols; t++) {
				B_by_i_over_t.col(t) = B[t].col(i).array()*mask.col(t).array();
			}
			if(!double_zero(b(i))){
				// take away the impact of the current b coefficient to the error matrix
				j_mask+=b(i)*B_by_i_over_t;
			}

			// perform only an update step if the current coefficient of Vtik2_vec is not zero
			if(!double_zero(Vtik2_vec[i])){
				// calculate rho (error matrix * B values / Vtik values)
				double rho=(j_mask * B_by_i_over_t).sum();

				// coordinate descent step to update 
				if(rho < - lambda_b){
					b(i)=(rho+lambda_b) / Vtik2_vec[i];
					j_mask-=b(i)*B_by_i_over_t;
				} else if ( rho > lambda_b){
					b(i)=(rho-lambda_b) / Vtik2_vec[i] ;
					j_mask-=b(i)*B_by_i_over_t;
				} else {
					b(i)=0.0;
					b_zero(i)=0;
				}
			} else {
				// set the coefficient to zero if the current coefficient of Vtik2_vec is zero
				b(i)=0.0;
				b_zero(i)=0;
			}
    	}
    }
}

/**
	* @brief: This function updates the linking matrix H in the coordinate descent algorithm.
	* It regularizes the parameters using lambda_H
	* @param M: matrix of observations
	* @param X: matrix of unit fixed effects
	* @param Z: matrix of time fixed effects
	* @param H: linking matrix
	* @param B: vector of unit-time covariates
	* @param X2Z2sum: matrix of X^2 + Z^2
	* @param mask: Treatment mask matrix for the current fold (1 if control and in training set, 0 otherwise)
	* @param L: matrix of lagged observations
	* @param u: vector of unit fixed effects
	* @param v: vector of time fixed effects
	* @param b: vector of unit-time covariate coefficients
	* @param num_rows: number of rows of the matrix of observations
	* @param num_cols: number of columns of the matrix of observations
	* @param H_rows: actual number of rows of the linking matrix after adding the identity matrix
	* @param H_cols: actual number of columns of the linking matrix after adding the identity matrix
	* @param H_rows_bef: number of rows of the linking matrix before adding the identity matrix
	* @param H_cols_bef: number of columns of the linking matrix before adding the identity matrix
	* @param to_estimate_u: boolean whether the unit fixed effects are estimated
	* @param to_estimate_v: boolean whether the time fixed effects are estimated
	* @param to_estimate_b: boolean whether the unit-time covariate coefficients are estimated
	* @param to_estimate_H: boolean whether the linking matrix is estimated
	* @param lambda_H: regularization parameter for the linking matrix
	* @return: void
	* @note: updating is in-place
*/
void update_H(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, MatrixXd &H, const std::vector<MatrixXd> &B, MatrixXd &X2Z2sum, const MatrixXd &mask, MatrixXd &L, VectorXd &u, VectorXd &v, VectorXd &b,int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, double lambda_H, MatrixXi &H_zero) {

	// sets the first linking matrix element to 0 to avoid any impact on the error matrix
	H(0,0)=0;
	ArrayXXd err_mat = Compute_err_Mat(M, mask, L, X, Z, H, B, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);

	// loop over the rows where the identity matrix is added. (bottom-right part of the linking matrix is 0)
	for(int i = 0; i < H_rows_bef; i++){
		for(int j = 0; j < H_cols; j++){
			//Do only run positions that have not been thresholded out
			if(H_zero(i,j)!=0){
				// Calculate the XZ matrix which definces the impact of the current coefficient on the outcome matrix
				ArrayXXd XZ = (X.col(i) * Z.row(j)).array() * mask.array();
				// Remove the impact of the current coefficient on the error matrix
				if(!double_zero(H(i,j))){
					err_mat+=H(i,j)*XZ;
				}

				// perform only an update step if the current coefficient of X2Z2sum is not zero
				if(!double_zero(X2Z2sum(i,j))){
					// calculate rho (error matrix * XZ values / X2Z2sum values)
					double rho =(XZ*err_mat).sum();
					// perform coordinate descent step to update the coefficient
					if(rho < - lambda_H){
						H(i,j)=(rho+lambda_H)/X2Z2sum(i,j);
						err_mat-=H(i,j)*XZ;
					} else if (rho > lambda_H){
						H(i,j)=(rho-lambda_H)/X2Z2sum(i,j);
						err_mat-=H(i,j)*XZ;
					} else {
						H(i,j)=0;
						H_zero(i,j)=0;
					}
				} else {
					// set the coefficient to zero if the current coefficient of X2Z2sum is zero
					H(i,j)=0;
					H_zero(i,j)=0;
				}
			}
		}
	}
	// loop over the bottom-left part of the H matrix
	if(H_rows>H_rows_bef){
		for(int i = H_rows_bef; i < H_rows; i++){
			for(int j = 0; j < H_cols_bef; j++){
				//Do only run positions that have not been thresholded out
				if(H_zero(i,j)!=0){
					// Calculate the XZ matrix which definces the impact of the current coefficient on the outcome matrix
					ArrayXXd XZ = (X.col(i) * Z.row(j)).array() * mask.array();
					if(!double_zero(H(i,j))){
						err_mat+=H(i,j)*XZ;
					}
					// perform only an update step if the current coefficient of X2Z2sum is not zero
					if(X2Z2sum(i,j)>0){
						// calculate rho (error matrix * XZ values / X2Z2sum values)
						double rho =(XZ*err_mat).sum()/X2Z2sum(i,j);
						// perform coordinate descent step to update the coefficient
						if(rho < - lambda_H){
							H(i,j)=rho+lambda_H;
							err_mat-=H(i,j)*XZ;
						} else if (rho > lambda_H){
							H(i,j)=rho-lambda_H;
							err_mat-=H(i,j)*XZ;
						} else {
							H(i,j)=0;
							H_zero(i,j)=0;
						}
					} else {
						// set the coefficient to zero if the current coefficient of X2Z2sum is zero
						H(i,j)=0;
						H_zero(i,j)=0;
					}
				}
			}
		}
	}
}

/**
 	* @brief: This function solves finds the optimal u and v assuming that L, H, and b are zero. This would be later
 	* helpful when we want to perform warm start on values of lambda_L and lambda_H. This function also outputs
	* the approximately smallest value of lambda_L and lambda_H which causes L, H, and b to be zero.
	* @param M: the outcome matrix
	* @param X: Units covariate matrix (Time-invariant)
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param B: Covariate matrix for the units (Time-invariant)
	* @param mask: Treatment mask matrix for the current fold (1 if control and in training set, 0 otherwise)
	* @param num_rows: number of rows in the outcome matrix
	* @param num_cols: number of columns in the outcome matrix
	* @param H_rows: actual number of rows of the linking matrix after adding the identity matrix
	* @param H_cols: actual number of columns of the linking matrix after adding the identity matrix
	* @param H_rows_bef: number of rows of the linking matrix before adding the identity matrix
	* @param H_cols_bef: number of columns of the linking matrix before adding the identity matrix
	* @param to_estimate_u: boolean indicating whether to estimate u or not
	* @param to_estimate_v: boolean indicating whether to estimate v or not
	* @param num_B_cov: number of unit-time covariates
	* @param niter: maximum number of iterations
	* @param rel_tol: relative tolerance for convergence
	* @return: a tuple containing the optimal u, v, lambda_L, lambda_H, lambda_b, X2Z2sum, and Vtik2_vec

*/
std::tuple<VectorXd, VectorXd, double, double, double, MatrixXd, std::vector<double>> initialize_uv(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, const std::vector<MatrixXd> &B, const MatrixXd &mask, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, int num_B_cov, int niter, double rel_tol) {

	// initialize the variables
    JacobiSVD<MatrixXd> svd(num_rows, num_cols);
    double obj_val = 0;
    double new_obj_val = 0;
    VectorXd u = VectorXd::Zero(num_rows);
    VectorXd v = VectorXd::Zero(num_cols);
    VectorXd b = VectorXd::Zero(num_B_cov);
    MatrixXd L = MatrixXd::Zero(num_rows, num_cols);
    MatrixXd H = MatrixXd::Zero(H_rows, H_cols);

    double sum_sigma=0.0;
    double lambda_L=0.0;
    double lambda_H=0.0;
    double lambda_b=0.0;

	// for the determination of the max lambda values, b and H are not estimated
    bool to_estimate_b=false;
    bool to_estimate_H=false;
    bool model_selection_H=false;
    bool model_selection_b=false;

	// initial objective function value
    obj_val = Compute_objval(M, X, Z, H, B, mask, L, u, v, b, sum_sigma, lambda_L, lambda_H, lambda_b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, model_selection_H, model_selection_b);
    for (int iter = 0; iter < niter; iter++) {
    	// iterate only over fixed effects u and v
		if (to_estimate_u) {
			update_u(M, X, Z, H, B, mask, L, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);
		}
		if (to_estimate_v) {
			update_v(M, X, Z, H, B, mask, L, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);
		}

		// Check if accuracy is achieved
		new_obj_val = Compute_objval(M, X, Z, H, B, mask, L, u, v, b, sum_sigma, lambda_L, lambda_H, lambda_b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, model_selection_H, model_selection_b);
		double rel_error = (obj_val - new_obj_val) / obj_val;
		if (std::abs(rel_error) < rel_tol) {
			break;
		}
		obj_val = new_obj_val;
    }

	// final value of objective function
    MatrixXd err_mat = Compute_err_Mat(M, mask, L, X, Z, H, B, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);

	// lambda_L is the maximum singular value of the error matrix divided by the number of observed elements
    svd.compute(err_mat);
    double lambda_L_max = svd.singularValues().maxCoeff();

    // generate the X2Z2 matrix that is afterwards used in the update step of H
    MatrixXd X2Z2sum = MatrixXd(H_rows, H_cols);
    double lambda_H_max=0;
    for (int i = 0; i < H_rows; i++) {
        for (int j = 0; j < H_cols; j++) {
        	if(i>H_rows_bef && j>H_cols_bef){
        		continue;
        	}
            ArrayXXd XZ = (X.col(i) * Z.row(j)).array() * mask.array();
            X2Z2sum(i,j) = (XZ * XZ).sum();
            double rho= (XZ*err_mat.array()).sum()/X2Z2sum(i,j);
            // in the same loop, calculate also the lambda_H_max as this depends on X2Z2
            if(abs(rho) > lambda_H_max){
				lambda_H_max=abs(rho);
			}
        }
    }

	// scale the lambda_H_max by the square root of the number of observed elements
    //lambda_H_max /= std::sqrt(mask.sum());

    // initialize the variables
    std::vector<double> Vtik2_vec;
    Vtik2_vec.reserve(num_B_cov);
    MatrixXd B_by_i_over_t = MatrixXd::Zero(num_rows, num_cols);
    double lambda_b_max=0;

	// loop over the number of covariates in B
	for (int i = 0; i < num_B_cov; i++) {
		// calculate the error matrix
		ArrayXXd err_mat = Compute_err_Mat(M, mask, L, X, Z, H, B, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);

		// Get the i-th column of each time-period of B
		for (int t = 0; t < num_cols; t++) {
			B_by_i_over_t.col(t) = B[t].col(i);
		}

		// calculate Vtik2 which is the sum of squares of the B_by_i_over_t matrix multiplied with the mask matrix.
		double Vtik2=(B_by_i_over_t.array() * B_by_i_over_t.array() * mask.array()).sum();
		Vtik2_vec.push_back(Vtik2);
		double rho=(err_mat * B_by_i_over_t.array()).sum() / Vtik2;
		// the lambda_max is the maximal value of the element-wise sum of the error matrix multiplied with the B_by_i_over_t matrix divided by Vtik2)
		if(abs(rho) > lambda_b_max){
			lambda_b_max=abs(rho);
		}
	}
	// scale the lambda_b_max by the square root of the number of observed elements
	// /= std::sqrt(mask.sum());

    return std::tuple<VectorXd, VectorXd, double, double, double, MatrixXd, std::vector<double>>(u,v,lambda_L_max, lambda_H_max, lambda_b_max, X2Z2sum, Vtik2_vec);
}

/**
 	* @brief: This function creates folds for cross-validation. Each fold contains a training and validation set.
 	* For each of these folds the initial solutions for fixed effects and the maximal lambda values are calculated.
 	* @param X: Units covariate matrix (Time-invariant)
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param B: Covariate matrix for the units (Time-invariant)
	* @param mask: Treatment mask matrix for the current fold (1 if control and in training set, 0 otherwise)
	* @param num_rows: Number of rows of the outcome matrix
	* @param num_cols: Number of columns of the outcome matrix
	* @param H_rows: actual number of rows of the linking matrix after adding the identity matrix
	* @param H_cols: actual number of columns of the linking matrix after adding the identity matrix
	* @param H_rows_bef: number of rows of the linking matrix before adding the identity matrix
	* @param H_cols_bef: number of columns of the linking matrix before adding the identity matrix
	* @param to_estimate_u: boolean indicating whether to estimate u or not
	* @param to_estimate_v: boolean indicating whether to estimate v or not
	* @param num_B_cov: number of unit-time covariates
	* @param niter: number of iterations
	* @param rel_tol: relative tolerance
	* @param cv_ratio: ratio of number of number of all observations that is used for cross validation
	* @param num_folds: number of folds for cross validation
	* @return: out: vector of tuples containing the initial solutions for fixed effects and the maximal lambda values for each fold, as well as the mask matrix for each fold
*/
std::vector<std::tuple<VectorXd, VectorXd, double, double, double, MatrixXd, std::vector<double>, MatrixXd>> create_folds(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, const std::vector<MatrixXd> &B, const MatrixXd &mask, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, int num_B_cov, int niter, double rel_tol, double cv_ratio, int num_folds, int seed) {

	// initiate variables
    std::vector<std::tuple<VectorXd, VectorXd, double, double, double, MatrixXd, std::vector<double>, MatrixXd>> out;
    //out.reserve(num_folds);
    double mask_array_sum=mask.array().sum();
    for (int k = 0; k < num_folds; k++) {
		std::default_random_engine generator(seed+k);
		std::bernoulli_distribution distribution(cv_ratio);
		ArrayXXd fold_mask=mask.array();
    	while(double_zero(fold_mask.sum()-mask_array_sum)){
    		// create a new random mask matrix for each fold
			MatrixXd ma_new(num_rows, num_cols);
			for (int i = 0; i < num_rows; i++) {
				for (int j = 0; j < num_cols; j++) {
					ma_new(i, j) = distribution(generator);
				}
			}
			fold_mask = mask.array() * ma_new.array();
    	}

        const MatrixXd fold_mask_final=fold_mask;
		// initialize u,v and get the maximal lambda values for each fold
        out.push_back(std::tuple_cat(initialize_uv(M, X, Z, B, fold_mask_final, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, num_B_cov, niter, rel_tol),std::tuple<MatrixXd>(fold_mask)));
    }
    return out;
}

/**
 	* @brief: This function performs cyclic coordinate descent updates. For given matrices M, mask, and initial starting decomposition given by L_init, H_init, u_init, v_init, and b_init
	* matrices L, H, u, and v are updated till convergence via coordinate descent.
	* @param M: Outcome matrix
	* @param X: Units covariate matrix (Time-invariant)
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param B: Covariate matrix for the units (Time-invariant)
	* @param H: Linking matrix
	* @param X2Z2sum: Sum of the squared columns of X and Z, used for the update of H
	* @param mask: Treatment mask matrix for the current fold (1 if control and in training set, 0 otherwise)
	* @param mask_validation: Validation mask matrix for the current fold (1 if control and in validation set, 0 otherwise)
	* @param L: rank-regularized matrix of unobserved factors
	* @param u: vector of unit fixed effects
	* @param v: vector of time fixed effects
	* @param b: parameter vector for the unit-time covariates
	* @param num_rows: Number of rows of the outcome matrix
	* @param num_cols: Number of columns of the outcome matrix
	* @param H_rows: actual number of rows of the linking matrix after adding the identity matrix
	* @param H_cols: actual number of columns of the linking matrix after adding the identity matrix
	* @param H_rows_bef: number of rows of the linking matrix before adding the identity matrix
	* @param H_cols_bef: number of columns of the linking matrix before adding the identity matrix
	* @param to_estimate_u: boolean indicating whether to estimate u or not
	* @param to_estimate_v: boolean indicating whether to estimate v or not
	* @param to_estimate_b: boolean indicating whether to estimate b or not
	* @param to_estimate_H: boolean indicating whether to estimate H or not
	* @param num_B_cov: number of unit-time covariates
	* @param niter: number of iterations
	* @param rel_tol: relative tolerance
	* @param lambda_L: regularization parameter for L
	* @param lambda_H: regularization parameter for H
	* @param lambda_b: regularization parameter for b
	* @param Vtik2_vec: vector of the division elements for the b updates
	* @param sing_restrict: vector of singular values that are set to 0
	* @param is_quiet: boolean indicating whether to print progress or not
	* @return: tuple containing L, H, u, v, b, MSE, and the final iteration number
*/
std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd, double> NNM_fit(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, const std::vector<MatrixXd> &B, MatrixXd H, MatrixXd &X2Z2sum, const MatrixXd &mask, MatrixXd L, VectorXd u, VectorXd v, VectorXd b, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, int num_B_cov, int niter, double rel_tol, double lambda_L, double lambda_H, double lambda_b, std::vector<double> Vtik2_vec, ArrayXd& sing_restrict, bool is_quiet, bool model_selection_H, bool model_selection_b) {

	// initialize variables
    double obj_val;
    double new_obj_val = 0;
    VectorXd sing;
    MatrixXd U;
    MatrixXd V;
    std::tie(sing,U,V)=MySVD(L);
    double sum_sigma = sing.sum();

    // matrices for thresholded out indices
	MatrixXi H_zero = MatrixXi::Ones(H_rows, H_cols);
	VectorXi b_zero = VectorXi::Ones(num_B_cov);

	// compute initial value of the objective function
    obj_val = Compute_objval(M, X, Z, H, B, mask, L, u, v, b, sum_sigma, lambda_L, lambda_H, lambda_b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, model_selection_H, model_selection_b);
    int finished_iter = 0;

	// perform cyclic coordinate descent updates
	for (int iter = 0; iter < niter; iter++) {

        // Update u
        if (to_estimate_u) {
            update_u(M, X, Z, H, B, mask, L, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);
        }
        // Update v
        if (to_estimate_v) {
            update_v(M, X, Z, H, B, mask, L, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);
        }
        // Update b
        if (to_estimate_b) {
            update_b(M, X, Z, H, B, mask, L, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, num_B_cov, lambda_b, Vtik2_vec, b_zero);
        }
        // Update H
        if (to_estimate_H) {
        	update_H(M, X, Z, H, B, X2Z2sum, mask, L, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, lambda_H, H_zero);
        }
        // Update L
        std::tie(L, sing) = update_L(M, X, Z, H, B, mask, L, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, sing_restrict, lambda_L);

        double sum_sigma = sing.sum();
        // Check if accuracy is achieved
        new_obj_val = Compute_objval(M, X, Z, H, B, mask, L, u, v, b, sum_sigma, lambda_L, lambda_H, lambda_b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, model_selection_H, model_selection_b);
        double rel_error = (obj_val - new_obj_val) / obj_val;
        finished_iter=iter;
        if ((std::abs(rel_error) < rel_tol) || double_zero(new_obj_val)) {
        	break;
        }
        obj_val = new_obj_val;
    }

	// get final MSE after iteration
    return std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd, double>(L, H, u, v, b, finished_iter);
}

/**
 	* @brief: Function that performs the NNM algorithm with initial values for L, H, u, v, and b for all the lamda configurations provided.
 	* @param M: outcome matrix
 	* @param L_init: initial value for matrix of unobserved factors
 	* @param X: Units covariate matrix (Time-invariant)
 	* @param H_init: initial value for the linking matrix between unit and time covariates
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param B: Covariate matrix for the units (Time-invariant)
	* @param X2Z2sum: Division elements for the update of update of H
	* @param mask: Treatment mask matrix for the current fold (1 if control and in training set, 0 otherwise)
	* @param mask_validation: Treatment mask matrix for the current fold (1 if control and in validation set, 0 otherwise)
	* @param u_init: initial value for the vector of unit-specific fixed effects
	* @param v_init: initial value for the vector of time-specific fixed effects
	* @param b_init: initial value for the vector of unit-specific covariate effects
	* @param lambda_tuples: vector of tuples containing the values of the regularization parameters
	* @param num_rows: number of rows of the outcome matrix
	* @param num_cols: number of columns of the outcome matrix
	* @param H_rows: actual number of rows of the linking matrix after the addition of the identity matrix
	* @param H_cols: actual number of columns of the linking matrix after the addition of the identity matrix
	* @param H_rows_bef: number of rows of the linking matrix before the addition of the identity matrix
	* @param H_cols_bef: number of columns of the linking matrix before the addition of the identity matrix
	* @param to_estimate_u: boolean indicating whether to estimate the unit-specific fixed effects
	* @param to_estimate_v: boolean indicating whether to estimate the time-specific fixed effects
	* @param to_estimate_b: boolean indicating whether to estimate the unit-specific covariate effects
	* @param to_estimate_H: boolean indicating whether to estimate the linking matrix
	* @param num_B_cov: number of covariates for the units
	* @param niter: maximum number of iterations
	* @param rel_tol: relative tolerance for the convergence of the algorithm
	* @param Vtik2_vec: vector of the division elements for the b updates
	* @param sing_restrict: vector of singular values that are set to 0
	* @param is_quiet: boolean indicating whether to print the progress of the algorithm
 	* @return: tuple containing the estimated matrices L, H, u, v, b, MSE, and the number of iterations
*/
std::vector<std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd, double>> NNM_with_uv_init(const MatrixXd &M, MatrixXd &L_init, const MatrixXd &X, MatrixXd &H_init, const MatrixXd &Z, const std::vector<MatrixXd> &B, MatrixXd &X2Z2sum, const MatrixXd &mask, VectorXd &u_init, VectorXd &v_init, VectorXd &b_init, std::vector<std::tuple<double, double, double>> lambda_tuples, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, int num_B_cov, int niter, double rel_tol, std::vector<double> Vtik2_vec, ArrayXd& sing_restrict, bool is_quiet, bool model_selection_H, bool model_selection_b) {

    int n_tuples = lambda_tuples.size();
    std::vector<std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd, double>> res(n_tuples);

	// Parallelization of the for loop that runs over all lambda configurations.
	#if defined(_OPENMP)
		#pragma omp parallel for schedule(dynamic)
	#endif
    for (int i = 0; i < n_tuples; i++) {
	    double lambda_H, lambda_L, lambda_b;
		std::tie(lambda_H, lambda_L, lambda_b)=lambda_tuples[i];
		// Call the NNM algorithm with the initial values for L, H, u, v, and b and the current lambda configuration.
		res[i] = NNM_fit(M, X, Z, B, H_init, X2Z2sum, mask, L_init, u_init, v_init, b_init, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, num_B_cov, niter, rel_tol, lambda_L, lambda_H, lambda_b, Vtik2_vec, sing_restrict, is_quiet, model_selection_H, model_selection_b);
    }
    return res;
}

/**
	* @brief: finds the optimal lambda configuration by cross-validation.
	* @param M: outcome matrix
	* @param X: Units covariate matrix (Time-invariant)
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param B: Covariate matrix for the units (Time-invariant)
	* @param mask: Overall treatment mask matrix (1 if control, 0 if treated)
	* @param num_rows: number of rows of the outcome matrix
	* @param num_cols: number of columns of the outcome matrix
	* @param H_rows: actual number of rows of the linking matrix after the addition of the identity matrix
	* @param H_cols: actual number of columns of the linking matrix after the addition of the identity matrix
	* @param H_rows_bef: number of rows of the linking matrix before the addition of the identity matrix
	* @param H_cols_bef: number of columns of the linking matrix before the addition of the identity matrix
	* @param to_estimate_u: boolean indicating whether to estimate the unit-specific fixed effects
	* @param to_estimate_v: boolean indicating whether to estimate the time-specific fixed effects
	* @param to_estimate_b: boolean indicating whether to estimate the unit-specific covariate effects
	* @param to_estimate_H: boolean indicating whether to estimate the linking matrix
	* @param num_B_cov: number of covariates for the units
	* @param niter: maximum number of iterations
	* @param rel_tol: relative tolerance for the convergence of the algorithm
	* @param cv_ratio: ratio of the observations used for the training set in the cross-validation
	* @param num_folds: number of folds in the cross-validation
	* @param is_quiet: boolean indicating whether to print the progress of the algorithm
	* @param cube_search: boolean indicating whether to perform a cube search for the optimal lambda configuration
	* @param n_config: number of configurations to be tested in the cube search
	* @param model_selection_H: Whether regularization on H should be performed
	* @param model_selection_b: Whether regularization on b should be performed
	* @param return_1se: Whether lambda values should be also evaluated for 1se optimality condition
	* @param seed: Random seed
	* @return: (tuple) returns a tuple f the following elements:
	L, H, u, v, b, min_lambda_L, min_lambda_H, min_lambda_b, min_MSE, MSE, lambda_Ls_est, lambda_Hs_est, lambda_bs_est
	* 			- L: estimated L matrix
	* 			- H: estimated H matrix
	* 			- u: estimated u vector
	* 			- v: estimated v vector
	* 			- b: estimated b vector
	* 			- min_lambda_L: optimal lambda_L value
	* 			- min_lambda_H: optimal lambda_H value
	* 			- min_lambda_b: optimal lambda_b value
	* 			- min_MSE: minimum MSE
	* 			- MSE: vector of the MSEs for the different lambda configurations
	* 			- lambda_Ls_est: vector of the lambda_L values tested
	* 			- lambda_Hs_est: vector of the lambda_H values tested
	* 			- lambda_bs_est: vector of the lambda_b values tested
	* 			- min_lambda_H_1se: optimal lambda_H value for 1se condition
	* 			- min_lambda_L_1se: optimal lambda_L value for 1se condition
	* 			- min_lambda_b_1se: optimal lambda_b value for 1se condition
*/
std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd, double, double, double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, double, double, double, double, MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd> Find_optimal_lambda(const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, const std::vector<MatrixXd> &B, const MatrixXd &mask, const MatrixXd &mask_null, int num_rows, int num_cols, int H_rows, int H_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, int num_B_cov, int niter, double rel_tol, double cv_ratio, int num_folds, bool is_quiet, bool cube_search, int n_config, bool model_selection_H, bool model_selection_b, bool return_1se, int seed, bool is_lambda_analysis, std::string file_path_cpp, int n_lambda) {

	if(!is_quiet){
		std::cout << " ---> Computing max lambda values" << std::endl;
	}
	// Create the folds. In each fold, initial values for u,v are determined by iteration only on u,v and max lambda values are calculated
	std::vector<std::tuple<VectorXd, VectorXd, double, double, double, MatrixXd, std::vector<double>, MatrixXd>>  confgs = create_folds(M, X, Z, B, mask_null, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, num_B_cov, niter, rel_tol, cv_ratio, num_folds, seed);

	//initialize all the variables
	double lam_L_max, lam_H_max, lam_b_max;
	double min_lambda_H_1se=0;
	double min_lambda_L_1se=0;
	double min_lambda_b_1se=0;

	std::vector<MatrixXd> mask_training_vec(num_folds);
	std::vector<MatrixXd> mask_validation_vec(num_folds);

	// initialize the matrices that will be updated in the algorithm
	VectorXd u, v, u_return, v_return;
	MatrixXd L, H, L_return, H_return;
	VectorXd b, b_return;
	bool b_all_zero=false;
	bool H_all_zero=false;
	bool L_all_zero=false;

	MatrixXd L_return_1se=MatrixXd::Zero(num_rows, num_cols);
	MatrixXd H_return_1se=MatrixXd::Zero(H_rows, H_cols);
	VectorXd b_return_1se=VectorXd::Zero(num_B_cov);
	VectorXd u_return_1se=VectorXd::Zero(num_rows);
	VectorXd v_return_1se=VectorXd::Zero(num_cols);

	double finished_iter;
	std::vector<VectorXd> u_vec(num_folds);
	std::vector<VectorXd> v_vec(num_folds);
	std::vector<VectorXd> b_vec(num_folds);
	std::vector<MatrixXd> L_vec(num_folds);
	std::vector<MatrixXd> H_vec(num_folds);

	ArrayXd sing_restrict = ArrayXd::Ones(std::min(num_rows, num_cols));

	std::vector<MatrixXd> X2Z2sum_vec(num_folds);
	std::vector<std::vector<double>> Vtik2_vec(num_folds);


	std::vector<double> max_lam_L(num_folds);
	std::vector<double> max_lam_H(num_folds);
	std::vector<double> max_lam_b(num_folds);

	// loop over the folds
	#if defined(_OPENMP)
		#pragma omp parallel for schedule(dynamic)
	#endif
	for (int k = 0; k < num_folds; k++) {

		VectorXd u_k, v_k;
		MatrixXd L_k, H_k;
		VectorXd b_k;
		double finished_iter_k;
		std::vector<double> Vtik2;
		MatrixXd X2Z2sum, mask_training;

		// extract the variables of the current fold and store them in vectors
		std::tie(u_k,v_k,lam_L_max, lam_H_max, lam_b_max, X2Z2sum, Vtik2, mask_training) = confgs[k];
		X2Z2sum_vec[k] = X2Z2sum;
		Vtik2_vec[k] = Vtik2;
		mask_training_vec[k] = mask_training;

		// create the validation mask
		MatrixXd mask_validation(num_rows, num_cols);
		mask_validation = mask_null.array() * (MatrixXd::Constant(num_rows, num_cols, 1.0) - mask_training).array();
		mask_validation_vec[k] = mask_validation;

		max_lam_L[k]=lam_L_max;
		max_lam_H[k]=lam_H_max;
		max_lam_b[k]=lam_b_max;

		MatrixXd L_init = MatrixXd::Zero(num_rows, num_cols);
		MatrixXd H_init = MatrixXd::Zero(H_rows, H_cols);
		VectorXd b_init = VectorXd::Zero(num_B_cov);

		// perform a once warm-start for L_init, H_init and b_init without regularization. The warm-start is performed on the first fold
		std::tie(L_k, H_k, u_k, v_k, b_k, finished_iter_k)=NNM_fit(M, X, Z, B, H_init, X2Z2sum, mask_training, L_init, u_k, v_k, b_init, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, num_B_cov, 2*niter, rel_tol, 0.0, 0.0, 0.0, Vtik2, sing_restrict, is_quiet, false, false);

		VectorXd sing;
		MatrixXd U;
		MatrixXd V;
		std::tie(sing,U,V)=MySVD(L_k);
		double sum_sigma = sing.sum();

		L_vec[k] = L_k;
		H_vec[k] = H_k;
		b_vec[k] = b_k;
		u_vec[k] = u_k;
		v_vec[k] = v_k;
	}
	// find the max lambda values over all folds

	double range_lam_L = *max_element(std::begin(max_lam_L), std::end(max_lam_L))/3;
	double range_lam_H = *max_element(std::begin(max_lam_H), std::end(max_lam_H))/3;
	double range_lam_b = *max_element(std::begin(max_lam_b), std::end(max_lam_b))/3;

	if(!is_quiet){
		std::cout << " ---> Max lambda H: " << 3*range_lam_H << std::endl;
		std::cout << " ---> Max lambda L: " << 3*range_lam_L << std::endl;
		std::cout << " ---> Max lambda b: " << 3*range_lam_b << std::endl << std::endl;
	}


	// initialize the variables for the loop
	double min_lambda_H=0;
	double min_lambda_L=0;
	double min_lambda_b=0;

	double min_MSE=0;
	int min_MSE_element=0;
	std::vector<double> MSE;
	MSE.reserve(n_config);
	std::vector<std::tuple<double, double, double>> configs_to_estimate;
	std::vector<std::tuple<double, double, double>> estimated_configs;
	estimated_configs.reserve(n_config+27);

	std::vector<double> finished_iter_vec;
	finished_iter_vec.reserve(n_config+27);

	int current_iter=0;
	int idx_H_no_cube=0;
	VectorXd lambda_Hs, lambda_Ls, lambda_bs;

	double old_min_lambda_L=range_lam_L;
	double old_min_lambda_H=range_lam_H;
	double old_min_lambda_b=range_lam_b;
	int n_lambda_H=1;
	int n_lambda_L=1;
	int n_lambda_b=1;
	double cv_se=0;

	if(!is_quiet){
		if(cube_search){
			std::cout << " -> Starting cube search." << std::endl;
		} else {
			std::cout << " -> Starting grid search." << std::endl;
		}
	}

	// loop over the number of configurations to estimate
	while(current_iter<(n_config-1)){
		//create initial set of lambdas for the first iteration
		if(current_iter==0){
			// if cube_search is true, the search is performed in a cube around the initial lambda values
			if(cube_search){
				configs_to_estimate.reserve(27);

				for(int idx_H=0; idx_H<=3; idx_H=idx_H*2+1){
					if(double_zero(range_lam_H)){
						idx_H=3;
					}
					for(int idx_L=0; idx_L<=3; idx_L=idx_L*2+1){
						if(double_zero(range_lam_L)){
							idx_L=3;
						}
						for(int idx_b=0; idx_b<=3; idx_b=idx_b*2+1){
							if(double_zero(range_lam_b)){
								idx_b=3;
							}
							configs_to_estimate.push_back(
									std::tuple<double, double, double>(idx_H*range_lam_H,
											idx_L*range_lam_L,
											idx_b*range_lam_b));
							if(!model_selection_b){idx_b=999;}
						}
					}
					if(!model_selection_H){idx_H=999;}
				}
			} else {
				// determine the number of lambda for the grid search
				if(model_selection_b){
					if(model_selection_H){
						n_lambda_b=std::ceil(std::cbrt((double)n_config));
						n_lambda_L=std::ceil(std::cbrt((double)n_config));
						n_lambda_H=std::ceil(std::cbrt((double)n_config));
					} else {
						n_lambda_b = std::ceil(std::sqrt((double)n_config));
						n_lambda_L = std::ceil(std::sqrt((double)n_config));
						n_lambda_H = 1;
					}
				} else {
					n_lambda_b=1;
					if(model_selection_H){
						n_lambda_L=std::ceil(std::sqrt((double)n_config));
						n_lambda_H=std::ceil(std::sqrt((double)n_config));
					} else {
						n_lambda_L = n_config;
						n_lambda_H = 1;
					}
				}

				// if cube_search is false, the search is performed on a grid
				configs_to_estimate.reserve(n_lambda_b*n_lambda_L);

				lambda_Hs=logsp(0.0, 3*range_lam_H, n_lambda_H);
				lambda_Ls=logsp(0.0, 3*range_lam_L, n_lambda_L);
				lambda_bs=logsp(0.0, 3*range_lam_b, n_lambda_b);

				for(int idx_L=0; idx_L<n_lambda_L; idx_L++){
					for(int idx_b=0; idx_b<n_lambda_b; idx_b++){
						configs_to_estimate.push_back(
								std::tuple<double, double, double>(lambda_Hs(idx_H_no_cube),
										lambda_Ls(idx_L),
										lambda_bs(idx_b)));
					}
				}
			}
		}

		// initialize vectors to store the results into
		int configs_to_estimate_size=configs_to_estimate.size();
		std::vector<double>current_MSE(configs_to_estimate_size,0);
		std::vector<std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd, double>> train_configs;
		train_configs.reserve(num_folds*configs_to_estimate_size);

		// loop over the folds
		std::vector<std::vector<double>> MSE_vec(configs_to_estimate_size);
		for(int k = 0; k < num_folds; k++){
			// train all the configurations
			std::vector<std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd, double>> train_configs_inner=NNM_with_uv_init(M, L_vec[k], X, H_vec[k], Z, B, X2Z2sum_vec[k], mask_training_vec[k], u_vec[k], v_vec[k], b_vec[k],
					configs_to_estimate, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, num_B_cov, niter, rel_tol, Vtik2_vec[k], sing_restrict, is_quiet, model_selection_H, model_selection_b);
			train_configs.insert(train_configs.end(), train_configs_inner.begin(), train_configs_inner.end());

			// loop over the results and extract the information of each configuration
			for(int i=0; i<configs_to_estimate_size; i++){
				std::tie(L, H, u, v, b, finished_iter) = train_configs_inner[i];

				finished_iter_vec.push_back(finished_iter);
				double MSE_config=Compute_MSE(M, X, Z, H, B, mask_validation_vec[k], L, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);
				current_MSE[i] += MSE_config/num_folds;
				MSE_vec[i].push_back(MSE_config);
			}
		}

		// find the minimum MSE position
		MSE.insert(MSE.end(), current_MSE.begin(), current_MSE.end());
		estimated_configs.insert(estimated_configs.end(), configs_to_estimate.begin(), configs_to_estimate.end());
		min_MSE_element=std::distance(std::begin(MSE), std::min_element(std::begin(MSE), std::end(MSE)));
		min_MSE=MSE[min_MSE_element];

		// update min_MSE, min_lambda, and the init values if necessary in case cube_search is true
		if(min_MSE_element>=current_iter){
			std::tie(min_lambda_H, min_lambda_L, min_lambda_b)=estimated_configs[min_MSE_element];
			min_MSE=MSE[min_MSE_element];
			L_return=MatrixXd::Zero(num_rows, num_cols);
			H_return=MatrixXd::Zero(H_rows, H_cols);
			b_return=VectorXd::Zero(num_B_cov);
			u_return=VectorXd::Zero(num_rows);
			v_return=VectorXd::Zero(num_cols);
			if(cube_search){
				for(int k=0; k<num_folds; k++){
					std::tie(L, H, u, v, b, finished_iter) = train_configs[k*configs_to_estimate_size+min_MSE_element-current_iter];
					L_vec[k]=L;
					L_return+=L/num_folds;
					H_vec[k]=H;
					H_return+=H/num_folds;
					b_vec[k]=b;
					b_return+=b/num_folds;
					u_vec[k]=u;
					u_return+=u/num_folds;
					v_vec[k]=v;
					v_return+=v/num_folds;
				}
				if(L_return.isZero(zero_tol)){ L_all_zero=true;} else { L_all_zero=false;}
				if(H_return.isZero(zero_tol)){ H_all_zero=true;} else { H_all_zero=false;}
				if(b_return.isZero(zero_tol)){ b_all_zero=true;} else { b_all_zero=false;}
				if(L_all_zero || H_all_zero || b_all_zero){
					std::transform(std::begin(MSE),std::end(MSE),std::begin(MSE),[](double x){return x+1;});
				}

			} else {
				for(int k=0; k<num_folds; k++){
					std::tie(L, H, u, v, b, finished_iter) = train_configs[k*configs_to_estimate_size+min_MSE_element-current_iter];
					L_return+=L/num_folds;
					H_return+=H/num_folds;
					b_return+=b/num_folds;
					u_return+=u/num_folds;
					v_return+=v/num_folds;
				}
			}
			cv_se=0;
			for(int k=0; k<num_folds; k++){
				cv_se+=1.0/(double(num_folds)-1.0)/(double)num_folds*std::pow(min_MSE-MSE_vec[min_MSE_element-current_iter][k],2);
			}
			cv_se=std::sqrt(cv_se);
		}
		// if cube_search is false, then use the estimate of the middle lambda configuration as new initial values
		if(!cube_search){
			for(int k=0; k<num_folds; k++){
				std::tie(L, H, u, v, b, finished_iter) = train_configs[k*configs_to_estimate_size+std::ceil(((double)n_lambda_L+0.5)/2*n_lambda_b)];
				L_vec[k]=L;
				H_vec[k]=H;
				b_vec[k]=b;
				u_vec[k]=u;
				v_vec[k]=v;
			}
		}

		current_iter+=configs_to_estimate_size;
		if(!is_quiet){
			std::cout << " ---> After " << current_iter << " configurations. The optimal lambdas are H: " << min_lambda_H << " , L: " << min_lambda_L << " , b: " << min_lambda_b << ", with MSE: " << min_MSE << std::endl;
		}

		// create the new configs to estimate
		if(current_iter<(n_config-1)){
			if(cube_search){

				// updating scheme here for lambda's
				if(double_zero(min_lambda_L)){
					range_lam_L/=6;
				} else if(min_lambda_L<old_min_lambda_L-range_lam_L/2){
					range_lam_L=std::min(range_lam_L/1.5, 2*min_lambda_L/3);
				} else if(min_lambda_L<old_min_lambda_L+range_lam_L/2){
					range_lam_L/=2;
				}
				if(double_zero(min_lambda_H)){
					range_lam_H/=6;
				} else if(min_lambda_H<old_min_lambda_H-range_lam_H/2){
					range_lam_H=std::min(range_lam_H/1.5, 2*min_lambda_H/3);
				} else if(min_lambda_H<old_min_lambda_H+range_lam_H/2){
					range_lam_H/=2;
				}
				if(double_zero(min_lambda_b)){
					range_lam_b/=6;
				} else if(min_lambda_b<old_min_lambda_b-range_lam_b/2){
					range_lam_b=std::min(range_lam_b/1.5, 2*min_lambda_b/3);
				} else if(min_lambda_b<old_min_lambda_b+range_lam_b/2){
					range_lam_b/=2;
				}
				old_min_lambda_H=min_lambda_H;
				old_min_lambda_L=min_lambda_L;
				old_min_lambda_b=min_lambda_b;

				if(!model_selection_H){range_lam_H=0;}
				if(!model_selection_b){range_lam_b=0;}

				configs_to_estimate.clear();
				for(int idx_H=-1; idx_H<=2; idx_H=(idx_H+1)*2){
					if(double_zero(range_lam_H)){
						idx_H=2;
					}
					if(!(idx_H==2 && H_all_zero)){
						for(int idx_L=-1; idx_L<=2; idx_L=(idx_L+1)*2){
							if(double_zero(range_lam_L)){
								idx_L=2;
							}
							if(!(idx_L==2 && L_all_zero)){
								for(int idx_b=-1; idx_b<=2; idx_b=(idx_b+1)*2){
									if(double_zero(range_lam_b)){
										idx_b=2;
									}
									if(!(idx_b==2 && b_all_zero)){
										double new_lambda_L=min_lambda_L+idx_L*range_lam_L;
										double new_lambda_H=min_lambda_H+idx_H*range_lam_H;
										double new_lambda_b=min_lambda_b+idx_b*range_lam_b;

										if(!(idx_L==0 && idx_H==0 && idx_b==0) &&
												!(double_zero(new_lambda_L) &&  double_zero(new_lambda_H) &&  double_zero(new_lambda_b)) &&
												(new_lambda_L>-zero_tol) &&
												(new_lambda_H>-zero_tol) &&
												(new_lambda_b>-zero_tol)){
											configs_to_estimate.push_back(
												std::tuple<double, double, double>(std::abs(new_lambda_H),
														std::abs(new_lambda_L),
														std::abs(new_lambda_b)));
										}

										if(idx_L==0 && idx_H==0 && idx_b==0 && (L_all_zero || H_all_zero || b_all_zero)){
											configs_to_estimate.push_back(
												std::tuple<double, double, double>(std::abs(new_lambda_H),
												std::abs(new_lambda_L),
												std::abs(new_lambda_b)));
										}
									}
								}
							}
						}
					}
				}
			} else {
				configs_to_estimate.clear();
				idx_H_no_cube++;
				for(int idx_L=0; idx_L<n_lambda_L; idx_L++){
					for(int idx_b=0; idx_b<n_lambda_b; idx_b++){
						configs_to_estimate.push_back(
								std::tuple<double, double, double>(lambda_Hs(idx_H_no_cube),
										lambda_Ls(idx_L),
										lambda_bs(idx_b)));
					}
				}
			}
		}

		if(configs_to_estimate.empty()){break;}
	}

	// retrieve the information at minimum MSE position
	std::tie(min_lambda_H, min_lambda_L, min_lambda_b)=estimated_configs[min_MSE_element];

	if (!is_quiet) {
		std::cout << std::endl << " -> Optimal lambda configuration on mse condition: " << std::endl;
		std::cout << " ---> lambda_H: " << min_lambda_H << std::endl;
		std::cout << " ---> lambda_L: " << min_lambda_L << std::endl;
		std::cout << " ---> lambda_b: " << min_lambda_b << std::endl;
		std::cout << " ---> Minimum MSE achieved on validation set: " << min_MSE << std::endl;
		std::cout << " ---> Standard error of cross-validation: " << cv_se << std::endl;
		double terminated_early= std::count_if(finished_iter_vec.begin(), finished_iter_vec.end(),[&](auto const& val){ return val <= niter-1.5; });
		std::cout << " ---> Share of early-terminated iteration procedures : " << round_to(terminated_early/finished_iter_vec.size(),3) << std::endl << std::endl;
	    size_t median_n = std::floor(finished_iter_vec.size() / 2);
	    nth_element(finished_iter_vec.begin(), finished_iter_vec.begin()+median_n, finished_iter_vec.end());
		std::cout << " ---> Median number of iteration procedures : " << finished_iter_vec[median_n] << std::endl << std::endl;
	}

	//Extract all the lambda configurations that have been estimated
	std::vector<double> lambda_Hs_est;
	lambda_Hs_est.reserve(MSE.size());
	std::vector<double> lambda_Ls_est;
	lambda_Ls_est.reserve(MSE.size());
	std::vector<double> lambda_bs_est;
	lambda_bs_est.reserve(MSE.size());
	double lambda_H, lambda_L, lambda_b;
	for(std::tuple<double, double, double> current_config : estimated_configs){
		std::tie(lambda_H, lambda_L, lambda_b)=current_config;
		lambda_Hs_est.push_back(lambda_H);
		lambda_Ls_est.push_back(lambda_L);
		lambda_bs_est.push_back(lambda_b);
	}

	if(return_1se){
		if(!is_quiet){
			std::cout << " -> Find optimal lambda configuration for 1se condition." << std::endl;
		}
		std::vector<std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd, double>> train_configs_outer;
		train_configs_outer.reserve(num_folds*num_1se_config);
		int outer_position;
		// distribute the se to all regularized spaces
		double se_div=(1.0+(double)model_selection_H+(double)model_selection_b);
		// generate the configurations to evaluate the L dimension
		std::vector<std::tuple<double, double, double>> configs_to_estimate_L;
		configs_to_estimate_L.reserve(num_1se_config);
		VectorXd lambda_Ls=logsp(min_lambda_L, *std::max_element(std::begin(max_lam_L), std::end(max_lam_L)), num_1se_config);
		for(int idx_L=0; idx_L<num_1se_config; idx_L++){
				configs_to_estimate_L.push_back(
						std::tuple<double, double, double>(min_lambda_H,lambda_Ls(idx_L),min_lambda_b));
		}

		// evaluate all the configurations
		std::vector<double> current_MSE_L(num_1se_config,0);
		std::vector<double> mse_0(num_folds);
		double mse_0_mean=0;
		double alternative_cv_se=0;
		for(int k = 0; k < num_folds; k++){
			// train all the configurations
			std::vector<std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd, double>> train_configs=NNM_with_uv_init(M, L_vec[k], X, H_vec[k], Z, B, X2Z2sum_vec[k], mask_training_vec[k], u_vec[k], v_vec[k], b_vec[k],
					configs_to_estimate_L, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, num_B_cov, niter, rel_tol, Vtik2_vec[k], sing_restrict, is_quiet, model_selection_H, model_selection_b);

			// loop over the results and extract the information of each configuration
			for(int i=0; i<num_1se_config; i++){
				std::tie(L, H, u, v, b, finished_iter) = train_configs[i];
				double MSE_config=Compute_MSE(M, X, Z, H, B, mask_validation_vec[k], L, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);
				current_MSE_L[i] += MSE_config/num_folds;
				if(i==0){
					mse_0[k]=MSE_config;
					mse_0_mean+=MSE_config/num_folds;
				}
			}
			train_configs_outer.insert(train_configs_outer.end(), train_configs.begin(), train_configs.end());
		}

		for(int k=0; k<num_folds; k++){
			alternative_cv_se+=1.0/(double(num_folds)-1.0)/(double)num_folds*std::pow(mse_0_mean-mse_0[k],2);
		}
		alternative_cv_se=std::sqrt(alternative_cv_se);

		double mse_plus_se_step=current_MSE_L[0]+alternative_cv_se/se_div;

		int position_L_1se = std::max(0,(int)std::distance(std::upper_bound(current_MSE_L.rbegin(), current_MSE_L.rend(), mse_plus_se_step, std::greater<double>()),current_MSE_L.rend())-1);
		min_lambda_L_1se=lambda_Ls(position_L_1se);
		outer_position=position_L_1se;

		if(model_selection_H){
			train_configs_outer.clear();
			train_configs_outer.reserve(num_folds*num_1se_config);
			// generate the configurations to evaluate the L dimension
			std::vector<std::tuple<double, double, double>> configs_to_estimate_H;
			configs_to_estimate_H.reserve(num_1se_config);
			VectorXd lambda_Hs=logsp(min_lambda_H, *std::max_element(std::begin(max_lam_H), std::end(max_lam_H)), num_1se_config);
			for(int idx_H=0; idx_H<num_1se_config; idx_H++){
				configs_to_estimate_H.push_back(
						std::tuple<double, double, double>(lambda_Hs(idx_H),
								min_lambda_L_1se,
								min_lambda_b));
			}

			// evaluate all the configurations
			std::vector<double> current_MSE_H(num_1se_config,0);
			mse_0_mean=0;
			alternative_cv_se=0;

			for(int k = 0; k < num_folds; k++){
				// train all the configurations
				std::vector<std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd, double>> train_configs=NNM_with_uv_init(M, L_vec[k], X, H_vec[k], Z, B, X2Z2sum_vec[k], mask_training_vec[k], u_vec[k], v_vec[k], b_vec[k],
						configs_to_estimate_H, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, num_B_cov, niter, rel_tol, Vtik2_vec[k], sing_restrict, is_quiet, model_selection_H, model_selection_b);

				// loop over the results and extract the information of each configuration
				for(int i=0; i<num_1se_config; i++){
					std::tie(L, H, u, v, b, finished_iter) = train_configs[i];
					double MSE_config=Compute_MSE(M, X, Z, H, B, mask_validation_vec[k], L, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);
					current_MSE_H[i] += MSE_config/num_folds;
					if(i==0){
						mse_0[k]=MSE_config;
						mse_0_mean+=MSE_config/num_folds;
					}
				}
				train_configs_outer.insert(train_configs_outer.end(), train_configs.begin(), train_configs.end());
			}

			for(int k=0; k<num_folds; k++){
				alternative_cv_se+=1.0/(double(num_folds)-1.0)/(double)num_folds*std::pow(mse_0_mean-mse_0[k],2);
			}
			alternative_cv_se=std::sqrt(alternative_cv_se);

			double mse_plus_se_step=current_MSE_H[0]+alternative_cv_se/se_div;

			int position_H_1se = std::max(0,(int)std::distance(std::upper_bound(current_MSE_H.rbegin(), current_MSE_H.rend(), mse_plus_se_step, std::greater<double>()),current_MSE_H.rend())-1);
			min_lambda_H_1se=lambda_Hs(position_H_1se);
			outer_position=position_H_1se;
		} else {
			min_lambda_H_1se=min_lambda_H;
		}

		if(model_selection_b){
			train_configs_outer.clear();
			train_configs_outer.reserve(num_folds*num_1se_config);
			// generate the configurations to evaluate the L dimension
			std::vector<std::tuple<double, double, double>> configs_to_estimate_b;
			configs_to_estimate_b.reserve(num_1se_config);
			VectorXd lambda_bs=logsp(min_lambda_b, *std::max_element(std::begin(max_lam_b), std::end(max_lam_b)), num_1se_config);
			for(int idx_b=0; idx_b<num_1se_config; idx_b++){
					configs_to_estimate_b.push_back(
							std::tuple<double, double, double>(min_lambda_H_1se,
									min_lambda_L_1se,
									lambda_bs(idx_b)));
			}
			// evaluate all the configurations
			std::vector<double> current_MSE_b(num_1se_config,0);
			mse_0_mean=0;
			alternative_cv_se=0;
			for(int k = 0; k < num_folds; k++){
				// train all the configurations
				std::vector<std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd, double>> train_configs=NNM_with_uv_init(M, L_vec[k], X, H_vec[k], Z, B, X2Z2sum_vec[k], mask_training_vec[k], u_vec[k], v_vec[k], b_vec[k],
						configs_to_estimate_b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, num_B_cov, niter, rel_tol, Vtik2_vec[k], sing_restrict, is_quiet, model_selection_H, model_selection_b);

				// loop over the results and extract the information of each configuration
				for(int i=0; i<num_1se_config; i++){
					std::tie(L, H, u, v, b, finished_iter) = train_configs[i];
					double MSE_config=Compute_MSE(M, X, Z, H, B, mask_validation_vec[k], L, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);
					current_MSE_b[i] += MSE_config/num_folds;
					if(i==0){
						mse_0[k]=MSE_config;
						mse_0_mean+=MSE_config/num_folds;
					}
				}
				train_configs_outer.insert(train_configs_outer.end(), train_configs.begin(), train_configs.end());
			}

			for(int k=0; k<num_folds; k++){
				alternative_cv_se+=1.0/(double(num_folds)-1.0)/(double)num_folds*std::pow(mse_0_mean-mse_0[k],2);
			}
			alternative_cv_se=std::sqrt(alternative_cv_se);

			double mse_plus_se_step=current_MSE_b[0]+alternative_cv_se/se_div;

			int position_b_1se = std::max(0,(int)std::distance(std::upper_bound(current_MSE_b.rbegin(), current_MSE_b.rend(), mse_plus_se_step, std::greater<double>()),current_MSE_b.rend())-1);
			min_lambda_b_1se=lambda_bs(position_b_1se);
			outer_position=position_b_1se;
		} else {
			min_lambda_b_1se=min_lambda_b;
		}

		if (!is_quiet) {
			std::cout << " ---> lambda_H_1se: " << min_lambda_H_1se << std::endl;
			std::cout << " ---> lambda_L_1se: " << min_lambda_L_1se << std::endl;
			std::cout << " ---> lambda_b_1se: " << min_lambda_b_1se << std::endl << std::endl;
		}

		for(int k=0; k<num_folds; k++){
			std::tie(L, H, u, v, b, finished_iter) = train_configs_outer[k*num_1se_config+outer_position];
			L_return_1se += L/num_folds;
			H_return_1se += H/num_folds;
			u_return_1se += u/num_folds;
			v_return_1se += v/num_folds;
			b_return_1se += b/num_folds;
		}
	}

	if(is_lambda_analysis){
		if(!is_quiet){
			std::cout << " -> Generate lambda configurations." << std::endl;
		}

		std::ofstream myfile_lambda_config;
		myfile_lambda_config.open(file_path_cpp+"/lambda_configs.csv");
		std::vector<std::tuple<double, double, double>> configs_to_estimate;
		configs_to_estimate.reserve(n_lambda*3+1);
		configs_to_estimate.push_back(std::tuple<double, double, double>(min_lambda_H, min_lambda_L, min_lambda_b));
		myfile_lambda_config << min_lambda_H << " " << min_lambda_L << " " << min_lambda_b << std::endl;

		double lam_L_max = *max_element(std::begin(lambda_Ls_est), std::end(lambda_Ls_est));
		double lam_H_max = *max_element(std::begin(lambda_Hs_est), std::end(lambda_Hs_est));
		double lam_b_max = *max_element(std::begin(lambda_bs_est), std::end(lambda_bs_est));

		VectorXd lambda_Hs=logsp(0.0, lam_H_max, n_lambda);
		VectorXd lambda_Ls=logsp(0.0, lam_L_max, n_lambda);
		VectorXd lambda_bs=logsp(0.0, lam_b_max, n_lambda);

		for(int i=0; i<n_lambda; i++){
			configs_to_estimate.push_back(std::tuple<double, double, double>(lambda_Hs(i),min_lambda_L,min_lambda_b));
			myfile_lambda_config << lambda_Hs(i) << " " << min_lambda_L << " " << min_lambda_b << std::endl;
			configs_to_estimate.push_back(std::tuple<double, double, double>(min_lambda_H,lambda_Ls(i),min_lambda_b));
			myfile_lambda_config << min_lambda_H << " " << lambda_Ls(i) << " " << min_lambda_b << std::endl;
			configs_to_estimate.push_back(std::tuple<double, double, double>(min_lambda_H,min_lambda_L,lambda_bs(i)));
			myfile_lambda_config << min_lambda_H << " " << min_lambda_L << " " << lambda_bs(i) << std::endl;
		}
		myfile_lambda_config.close();

		if(!is_quiet){
			std::cout << " ---> Estimate CV results with grid on optimal lambda values." << std::endl;
		}

		std::ofstream myfile_cv_config;
		myfile_cv_config.open(file_path_cpp+"/CV_configs.csv");
		// Create the folds. In each fold, initial values for u,v are determined by iteration only on u,v and max lambda values are calculated
		for (int k = 0; k < num_folds; k++) {
			VectorXd u_k, v_k;
			MatrixXd L_k, H_k;
			VectorXd b_k;
			double finished_iter_k;
			ArrayXd sing_restrict_k = ArrayXd::Ones(std::min(num_rows, num_cols));
			std::vector<std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd, double>> lambda_configs_k =NNM_with_uv_init(M, L_return, X, H_return, Z, B, X2Z2sum_vec[k], mask_training_vec[k], u_return, v_return, b_return,
					configs_to_estimate, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, num_B_cov, niter, rel_tol, Vtik2_vec[k], sing_restrict_k, is_quiet, model_selection_H, model_selection_b);

			for(int i=0; i<configs_to_estimate.size(); i++){
				std::tie(L_k, H_k, u_k, v_k, b_k, finished_iter_k) = lambda_configs_k[i];
				double MSE_config=Compute_MSE(M, X, Z, H_k, B, mask_validation_vec[k], L_k, u_k, v_k, b_k, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H);
				myfile_cv_config << MSE_config << " ";
			}
			myfile_cv_config << std::endl;
		}
		myfile_cv_config.close();
	}
	return(std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd, double, double, double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, double, double, double, double, MatrixXd, MatrixXd, VectorXd, VectorXd, VectorXd>(L_return, H_return, u_return, v_return, b_return, min_lambda_L, min_lambda_H, min_lambda_b, min_MSE, MSE, lambda_Ls_est, lambda_Hs_est, lambda_bs_est, min_lambda_H_1se, min_lambda_L_1se, min_lambda_b_1se, cv_se, L_return_1se, H_return_1se, u_return_1se, v_return_1se, b_return_1se));
}

/**
	* @brief: re-estimate the parameters without regularization but only on the non-zero entries of the matrices.
	* @param L: rank-regularized matrix of unobserved factors
	* @param H: Linking matrix for unit and time covariates
	* @param u: Unit fixed effects
	* @param v: Time fixed effects
	* @param b: Unit-Time covariate coefficients
	* @param M: Outcome matrix
	* @param X: Units covariate matrix (Time-invariant)
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param B_vec: Vector of Unit-Time covariate matrices
	* @param mask: Overall treatment mask matrix (1 if control, 0 if treated)
	* @param X2Z2sum: Division elements for the update of H
	* @param Vtik2_vec: Vector of division elements for the update of b
	* @param num_B_cov: Number of Unit-Time covariates
	* @param H_rows: actual number of rows of the linking matrix after the addition of the identity matrix
	* @param H_cols: actual number of columns of the linking matrix after the addition of the identity matrix
	* @param num_rows: number of rows of the outcome matrix
	* @param num_cols: number of columns of the outcome matrix
	* @param H_rows_bef: number of rows of the linking matrix before the addition of the identity matrix
	* @param H_cols_bef: number of columns of the linking matrix before the addition of the identity matrix
	* @param to_estimate_u: boolean indicating whether to estimate the unit fixed effects
	* @param to_estimate_v: boolean indicating whether to estimate the time fixed effects
	* @param to_estimate_b: boolean indicating whether to estimate the unit-time covariate coefficients
	* @param to_estimate_H: boolean indicating whether to estimate the linking matrix
	* @param niter: number of iterations
	* @param rel_tol: relative tolerance for the stopping criterion
	* @param is_quiet: boolean indicating whether to print the progress of the algorithm
	* @return: void
	* @note: the function updates the parameters in place
*/
void eval_post(MatrixXd &L, MatrixXd &H, VectorXd &u, VectorXd &v, VectorXd &b, const MatrixXd &M, const MatrixXd &X, const MatrixXd &Z, const std::vector<MatrixXd> &B_vec, const MatrixXd &mask, MatrixXd X2Z2sum, std::vector<double> Vtik2_vec, int num_B_cov, int H_rows, int H_cols, int num_rows, int num_cols, int H_rows_bef, int H_cols_bef, bool to_estimate_u, bool to_estimate_v, bool to_estimate_b, bool to_estimate_H, int niter, double rel_tol, bool is_quiet){

	// sets values in the Vtik2 matrix to zero below the double precision threshold
	for (int i = 0; i < num_B_cov; i++) {
		if (double_zero(b(i))){
			Vtik2_vec[i]=0.0;
		}
	}

	// sets values in the X2Z2sum matrix to zero below the double precision threshold
	for (int i = 0; i < H_rows; i++) {
		for (int j = 0; j < H_cols; j++) {
			if(double_zero(H(i,j))){
				X2Z2sum(i,j) = 0.0;
			}
		}
	}
	// initialize the SVD of L
	MatrixXd U;
	MatrixXd V;
	VectorXd sing;

	std::tie(sing, U, V) = MySVD(L);
	ArrayXd sing_restrict(sing.size());
	for(int i=0; i<sing.size(); i++){
		sing_restrict(i)=(double) !double_zero(sing(i));
	}

	double finished_iter;
	// run the NNM algorithm without regularization
	std::tie(L, H, u, v, b, finished_iter)= NNM_fit(M, X, Z, B_vec, H, X2Z2sum, mask, L, u, v, b, num_rows, num_cols, H_rows, H_cols, H_rows_bef, H_cols_bef, to_estimate_u, to_estimate_v, to_estimate_b, to_estimate_H, num_B_cov, 2*niter, rel_tol, 0.0, 0.0, 0.0, Vtik2_vec, sing_restrict, is_quiet, false, false);
}
