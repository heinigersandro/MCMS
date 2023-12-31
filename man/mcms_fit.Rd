% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mc_ms.R
\name{mcms_fit}
\alias{mcms_fit}
\title{This function is used to estimate the MCNNM model with given lambda values for L,H, and b.
If post_estimation is set, then the estimated parameters are returned additionally in the list.}
\usage{
mcms_fit(
  M,
  X,
  Z,
  B,
  mask,
  lambda_L,
  lambda_H,
  lambda_b,
  num_B_cov,
  to_add_ID = TRUE,
  to_estimate_u = TRUE,
  to_estimate_v = TRUE,
  to_estimate_b = TRUE,
  to_estimate_H = TRUE,
  niter = 100,
  rel_tol = 1e-05,
  to_normalize = TRUE,
  is_quiet = FALSE,
  post_estimation = TRUE
)
}
\arguments{
\item{M}{Outcome matrix}

\item{X}{Units covariate matrix (Time-invariant)}

\item{Z}{Time covariate matrix (Unit-invariant)}

\item{B}{Unit-Time covariate matrix}

\item{mask}{Treatment mask matrix (1 if control, 0 if treated)}

\item{lambda_L}{lambda value for rank-regularization}

\item{lambda_H}{lambda value for unit and time covariate linking matrix regularization}

\item{lambda_b}{lambda value for unit-time covariate regularization}

\item{num_B_cov}{Number of unit-time covariates}

\item{to_add_ID}{Whether to add an identity matrix to the X and Z matrices. (default: TRUE)}

\item{to_estimate_u}{Whether to estimate the units' fixed effects. (default: TRUE)}

\item{to_estimate_v}{Whether to estimate the time's fixed effects. (default: TRUE)}

\item{to_estimate_b}{Whether to estimate the unit-time varying covariate parameters. (default: TRUE)}

\item{to_estimate_H}{Whether to estimate the unit and time covariates linking parameters. (default: TRUE)}

\item{niter}{Number of iterations for the coordinate descent steps. (default: 100)}

\item{rel_tol}{Relative tolerance for the convergence of the coordinate descent steps. (default: 1e-5)}

\item{to_normalize}{Whether to normalize the X, Z, and B matrices. (default: TRUE)}

\item{is_quiet}{Whether to print the progress of the algorithm. (default: FALSE)}

\item{post_estimation}{Whether a post regularization estimation without regularization should be estimated}
}
\value{
: List containing the following elements:
	    - "u": Estimated units' fixed effects
		- "v": Estimated time's fixed effects
		- "B": Estimated unit-time covariates' coefficients
		- "H": Estimated H matrix
		- "tau": The estimate treatment effect
		- "lambda_L": Estimated lambda_L value
		- "lambda_H": Estimated lambda_H value
		- "lambda_b": Estimated lambda_b value
}
\description{
This function is used to estimate the MCNNM model with given lambda values for L,H, and b.
If post_estimation is set, then the estimated parameters are returned additionally in the list.
}
\examples{
mcms_fit(M, X, Z, B, mask, lambda_L, lambda_H, lambda_b, num_B_cov)
}
