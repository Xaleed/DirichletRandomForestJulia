# DirichletRandomForest

This repository provides Julia implementations of Random Forest algorithms based on the Dirichlet distribution.

## Initialization

The core Julia scripts are located in the `src` folder:

- `dirichlet_forest_ml.jl`
- `dirichlet_forest_ml_distributed.jl`
- `MLE_vs_MoM.jl`

## Using from R

You can also use these Julia functions from within R. To get started, install Julia and set up the `JuliaCall` package in R:

```r
if (!require("JuliaCall")) install.packages("JuliaCall")
library(JuliaCall)
install_julia()
