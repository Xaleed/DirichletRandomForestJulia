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
```
The main function for calling the Julia code from R is defined in the `dirichlet_forest_wrapper.jl` file located in the src folder.
To see how to use it in practice, refer to the `example_usage_dirichlet_forest.R` file in the examples folder.
