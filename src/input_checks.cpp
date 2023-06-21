

#include <cmath>
#include <vector>
#include <tuple>

#include <Rcpp.h>
#include <RcppEigen.h>
#include <stdlib.h>

#include "input_checks.h"
#include "update_functions.h"

using namespace Eigen;
using namespace Rcpp;

/**
    * @brief: Checks if the mask matrix is a valid binary matrix
    * @param mask: the matrix of missing values
    * @return: a boolean indicating whether the mask matrix is a valid binary matrix
*/
bool mask_check(const MatrixXd &mask) {
    MatrixXd H = (MatrixXd::Constant(mask.rows(), mask.cols(), 1.0) - mask).array() * mask.array();
    return(H.isZero(1e-5));
}

/**
    * @brief: Checks if the X matrix has the correct dimensions
    * @param M: Outcome matrix
	* @param X: Units covariate matrix (Time-invariant)
    * @return: a boolean indicating whether the X matrix has the correct dimensions
*/
bool X_size_check(const MatrixXd &M, MatrixXd &X) {
    return (M.rows() == X.rows());
}

/**
    * @brief: Checks if the Z matrix has the correct dimensions
    * @param M: Outcome matrix
    * @param Z: Time covariate matrix (Time-varying)
    * @return: a boolean indicating whether the Z matrix has the correct dimensions
*/
bool Z_size_check(const MatrixXd &M, MatrixXd &Z) {
    return (M.cols() == Z.rows());
}

/**
    * @brief: Checks if the B matrix has the correct dimensions
    * @param M: Outcome matrix
	* @param B: Unit-Time covariate matrix
    * @return: a boolean indicating whether the B matrix has the correct dimensions
*/
bool mask_size_check(const MatrixXd &M, const MatrixXd &mask) {
    return (M.rows() == mask.rows() && M.cols() == mask.cols());
}

/**
    * @brief: Normalizes the columns of a matrix in-place and returns the norms of the columns.
    * Is used for the normalization of the X and Z matrices.
    * @param mat: the matrix to be normalized
    * @return: a vector containing the norms of the columns of the matrix
*/
VectorXd normalize(MatrixXd &mat) {
    VectorXd col_norms = VectorXd::Zero(mat.cols());
    if (mat.cols() > 0) {
        for (int i = 0; i < mat.cols(); i++) {
        	double col_norms_i=mat.col(i).norm();
            col_norms(i) = col_norms_i;
            mat.col(i) /= col_norms_i;
        }
    }
    return col_norms;
}

/** 
    * @brief: Normalizes the columns of a matrix in-place and returns the norms of the columns.
    * Is used for the normalization of the B matrix as it takes into account the special structurre of B. 
    * @param mat: the matrix to be normalized
    * @param num_B_cov: the number of covariates in B
    * @param num_rows: the number of rows if the outcome matrix
    * @param num_cols: the number of columns if the outcome matrix
    * @return: a vector containing the norms of the columns of the matrix
    * @note: the normalization is done by taking all columns with a distance of num_B_cov from each other and normalizing them together.
    * This is done according to the special structure of B.
    * The normalization is done by first creating a vector with all the columns of B and then normalizing this vector.
    * The vector is then split up again into the columns of B.
    * This is done to solve the problem of normalizing all columns representing the same variable with the same normalization factor.
*/
VectorXd normalize_B(MatrixXd &mat, int num_B_cov, int num_rows, int num_cols) {

    VectorXd col_norms = VectorXd::Zero(num_B_cov);
    if (mat.cols() > 0) {
        for (int i = 0; i < num_B_cov; i++) {
        	VectorXd full_vector=VectorXd::Zero(num_rows*num_cols);
        	for(int j = 0; j < num_cols; j++){
        		VectorXd to_copy=VectorXd::Zero(num_rows);
        		to_copy=mat.col(j*num_B_cov+i);
				for(int k = 0; k < num_rows; k++){
					full_vector(j*num_rows+k)=to_copy(k);
				}
        	}
        	double col_norm=full_vector.norm();
        	col_norms(i) = col_norm;
        	for(int j = 0; j < num_cols; j++){
        		mat.col(j*num_B_cov+i) /=  col_norm;
        	}
        }
    }
    return col_norms;
}

/**
    * @brief: Re-normalizes the rows of a matrix according to the vector of row-norms and returns the matrix.
    * Is used for the re-normalization of the H matrix.
    * @param H: the matrix to be re-normalized
    * @param row_H_scales: the vector containing the row-norms
    * @return: the re-normalized matrix
*/
MatrixXd normalize_back_rows(MatrixXd H, VectorXd &row_H_scales) {
    for (int i = 0; i < row_H_scales.size(); i++) {
		H.row(i) /= row_H_scales(i);
	}
    return(H);
}

/**
    * @brief: Re-normalizes the columns of a matrix according to the vector of column-norms and returns the matrix.
    * Is used for the re-normalization of the H matrix.
    * @param H: the matrix to be re-normalized
    * @param col_H_scales: the vector containing the column-norms
    * @return: the re-normalized matrix
*/
MatrixXd normalize_back_cols(MatrixXd H, VectorXd &col_H_scales) {
	for (int i = 0; i < col_H_scales.size(); i++) {
		H.col(i) /= col_H_scales(i);
	}
    return(H);
}

/**
    * @brief: Re-normalizes the vector according to the vector of column-norms and returns the vector.
    * Is used for the re-normalization of the beta vector.
    * @param vec: the vector to be re-normalized
    * @param vec_scales: the vector containing the column-norms of B
    * @return: the re-normalized vector
*/
VectorXd normalize_back_vector(VectorXd vec, VectorXd &vec_scales) {
	ArrayXd vec_array=vec.array();
	ArrayXd vec_scales_array=vec_scales.array();
	VectorXd res = vec_array/vec_scales_array;
    return(res);
}

/**
    * @brief: Transforms the B matrix into a vector of matrices. Each vector element represents on time-period.
    * @param B: the matrix to be transformed
    * @param num_B_cov: the number of covariates in B
    * @return: a vector of matrices
    * @note: The transformation is done by repeatedly taking a subsequent chunk of num_B_cov columns of B and transforming them into a matrix.
*/
const std::vector<MatrixXd> vectorize_B(MatrixXd &B, int num_B_cov) {

    int num_cols = B.cols();
    int num_rows = B.rows();
    int vec_entries = std::floor(((float)num_cols + (0.1 / (float)num_B_cov)) / (float)num_B_cov);

    std::vector<MatrixXd> B_vec(vec_entries);

    for (int i = 0; i < vec_entries; ++i) {
        Map<Eigen::MatrixXd> B_i(B.data() + i * num_rows * num_B_cov, num_rows, num_B_cov);
        MatrixXd B_i_(B_i);
        B_vec.at(i) = B_i_;
    }
    const std::vector<MatrixXd> B_vec_final(B_vec);
    return B_vec_final;
}

/**
    * @brief: Checks all the matrix dimensions, whether covariate matrices are supplied, and whether the mask matrix is valid.
    * @param M: The outcome matrix
    * @param X: Units covariate matrix (Time-invariant)
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param B: Unit-Time covariate matrix
	* @param mask: Treatment mask matrix
	* @param to_add_ID: Whether to add an identity matrix to X and Z
    * @return: true if all checks are passed, false otherwise
*/
bool mcnnm_matrix_check(const MatrixXd &M, MatrixXd &X, MatrixXd &Z, MatrixXd &B, const MatrixXd &mask, bool to_add_ID) {

    if (!mask_check(mask)) {
        std::cerr << "Error: The mask matrix should only include 0 (for missing) and 1 (for observed entries)" << std::endl;
        return false;
    }
    if (!mask_size_check(M, mask)) {
        std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
        return false;
    }

    if (X.rows() == 0 && !to_add_ID) {
        std::cerr << "Error: No need for training H as X is empty and identity matrix addition is disabled. Run mcnnm_lam_range instead" << std::endl;
        return false;
    }

    if (Z.rows() == 0 && !to_add_ID) {
        std::cerr << "Error: No need for training H as Z is empty and identity matrix addition is disabled. Run mcnnm_lam_range instead" << std::endl;
        return false;
    }

    if (X.rows() == 0 && Z.rows() == 0) {
        std::cerr << "Error: No need for training H as X and Z are both empty. Run mcnnm_lam_range instead" << std::endl;
        return false;
    }

    if (X.rows() > 0 && !X_size_check(M, X)) {
        std::cerr << "Error: Number of rows of X should match with the number of rows of M" << std::endl;
        return false;
    }
    if (Z.rows() > 0 && !Z_size_check(M, Z)) {
        std::cerr << "Error: Number of rows of Z should match with the number of columns of M" << std::endl;
        return false;
    }
    return true;
}

/**
    * @brief: Checks the chosen rel_tol value. If it is very small, a warning is printed.
    * @param rel_tol: The chosen relative improvement value
    * @return: true if the check is passed, false otherwise
    * @note: The relative improvement value is used to determine whether the training should stop. If the relative improvement is smaller than the chosen value, the training stops.
    * The smaller the value, the longer the training takes.
    * The default value is 1e-6.
    * If the value is smaller than 1e-10, a warning is printed.
*/
bool mcnnm_tol_check(double rel_tol) {

    if (rel_tol < 1e-10) {
        std::cerr << "Warning: The chosen value for relative improvement is very small. Training might take longer" << std::endl;
    }
    return true;
}

/**
    * @brief: Checks the chosen values of n_config, cv_ratio, and num_folds.
    * If the n_config value is above 2500, a warning is printed.
    * If the cv_ratio value is not between 0.1 and 0.9, an error is printed.
    * If the num_folds value is not between 2 and 10, an error is printed.
    * @param n_config: The number of configs of lambda that are evaluated to find the optimal value.
    * @param cv_ratio: The ratio of the data that is used for cross-validation.
    * @param num_folds: The number of random folds that are used for cross-validation.
    * @return: true if the check is passed, false otherwise
    * @note: 
    * The default values are 1000, 0.2, and 5, respectively.
*/
bool mcnnm_cv_check(int n_config, double cv_ratio, int num_folds) {

    if (n_config > 2500) {
        std::cerr << "Warning: The cross-validation might take very long. Please decrease number of configurations that are evaluated" << std::endl;
    }
    if (cv_ratio < 0.1 || cv_ratio > 0.9) {
        std::cerr << "Error: The cross-validation ratio should be between 10 to 90 percent for getting accurate results. Please modify it" << std::endl;
        return false;
    }

    if (num_folds > 5) {
        std::cerr << "Warning: Number of random folds are chosen to be greater than 5. This process might take long" << std::endl;
    }
    return true;
}

/**
    * @brief: Adds the identity matrix to the right of X.
    * @param X: The matrix to which the identity matrix is added
    * @param num_rows: The number of rows of the identity matrix, must match the rows of X
    * @return: The concatenated matrix
*/
MatrixXd X_add_id(MatrixXd X, int num_rows){
	MatrixXd X_add = MatrixXd::Identity(num_rows, num_rows);
	MatrixXd X_conc(num_rows, X.cols() + X_add.cols());
	X_conc << X, X_add;
	return X_conc;
}

/**
    * @brief: Adds the identity matrix to the bottom of Z.
    * @param Z: The matrix to which the identity matrix is added
    * @param num_cols: The number of cols of the identity matrix, must match the cols of Z 
    * @return: The concatenated matrix
*/
MatrixXd Z_add_id(MatrixXd Z, int num_cols){
	MatrixXd Z_add = MatrixXd::Identity(num_cols, num_cols);
	MatrixXd Z_conc(num_cols, Z.cols() + Z_add.cols());
	Z_conc << Z, Z_add;
	return Z_conc;
}

/**
    * @brief: Prepares the input data for training.
    * First it performs the necessary checks on the input data.
    * Then it adds the identity matrix to X and Z if chosen by to_add_id.
    * Finally, it normalizes the data if chosen by to_normalize. 
    * The column norms of X, Z, and B are stored in X_col_norms, Z_col_norms, and B_col_norms, respectively.
    * @param M: The outcome matrix
    * @param X: Units covariate matrix (Time-invariant)
	* @param Z: Time covariate matrix (Unit-invariant)
	* @param B: Unit-Time covariate matrix
	* @param mask: Treatment mask matrix
    * @param to_add_ID: Whether to add the identity matrix to X and Z
    * @param num_rows: The number of rows of the outcome matrix
    * @param num_cols: The number of columns of the outcome matrix
    * @param X_col_norms: Vector where the column norms of X are stored
    * @param Z_col_norms: Vector where the column norms of Z are stored
    * @param B_col_norms: Vector where the column norms of B are stored
    * @param to_normalize: Boolean whether to normalize the data
    * @param num_B_cov: The number of unit-time covariates per time-period
    * @return: void
*/
void prepare_data(const MatrixXd &M, MatrixXd &X, MatrixXd &Z, MatrixXd &B, const MatrixXd &mask, int to_add_ID, int num_rows, int num_cols, VectorXd &X_col_norms, VectorXd &Z_col_norms, VectorXd &B_col_norms, bool to_normalize, int num_B_cov){

	bool input_checks = mcnnm_matrix_check(M, X, Z, B, mask, to_add_ID);

	if (!input_checks) {
		throw std::invalid_argument("Invalid inputs ! Please modify");
	}

	if(to_add_ID==1){
		X=X_add_id(X, num_rows);
		Z=Z_add_id(Z, num_cols);
	}

	if (to_normalize && X.cols() > 0) {
		X_col_norms = normalize(X);
	}
	if (to_normalize && Z.cols() > 0) {
		Z_col_norms = normalize(Z);
	}
	if (to_normalize && B.cols() > 0) {
		B_col_norms = normalize_B(B, num_B_cov, num_rows, num_cols);
	}

}


