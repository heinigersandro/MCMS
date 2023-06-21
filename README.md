# MCMS

Matrix completion method for causal panel data models with data-driven model selection

The __MCMS__ package provides functions to calculate the average treatment effect on the treated (ATET) estimating a low-rank factor model plus noise with a linear covariate term. The method usese nuclear norm regularization on the rank of the matrix of unobserved factors and l-1 regularization on the covariate model to simultaneously reduce the model complexity in the covariates.

To install this package in R, run the following commands:

```R
install.packages("devtools")
library(devtools) 
install_github("sandroheiniger/MCMS")
```

#### Version

0.1.0 : Initial version of the estimator. 


#### References
Susan Athey, Mohsen Bayati, Nikolay Doudchenko, Guido Imbens, and Khashayar Khosravi. <b>Matrix Completion Methods for Causal Panel Data Models</b> [<a href="http://arxiv.org/abs/1710.10251">link</a>]