# MCMS

Matrix completion method for causal panel data models with data-driven model selection

The __MCMS__ package provides functions to calculate the average treatment effect on the treated (ATET), estimating a low-rank factor model plus noise with a linear covariate term using the matrix completion method. The method uses nuclear norm regularization on the rank of the matrix of unobserved factors and l-1 regularization on the covariate model to simultaneously reduce the model complexity in the covariates.

To install this package in R, run the following commands:

```R
install.packages("devtools")
library(devtools) 
install_github("heinigersandro/MCMS")
```

#### Version

0.1.0 : Initial version of the estimator. 
1.0.0 : Stable version

#### References
Susan Athey, Mohsen Bayati, Nikolay Doudchenko, Guido Imbens, and Khashayar Khosravi. <b>Matrix Completion Methods for Causal Panel Data Models</b> [<a href="https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1891924">link</a>]
Victor Chernozhukov, Kaspar Wüthrich, and Yinchu Zhu. <b>An exact and robust conformal inference method for counterfactual and synthetic controls</b> [<a href="https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1920957">link</a>]