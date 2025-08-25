# DirichletRandomForest

This repository provides Julia implementations of Random Forest algorithms based on the Dirichlet distribution.

## Initialization

The core Julia scripts are located in the `src` folder:

- `dirichlet_forest_ml.jl`
- `dirichlet_forest_ml_distributed.jl`
- `MLE_vs_MoM.jl`

### About `q_threshold` (Number of Quantiles for Splitting)

In the following function signature:

```julia
fit_dirichlet_forest!(
    forest::DirichletForest,
    X::Matrix{Float64},
    Y::Matrix{Float64},
    q_threshold::Int = 500000000,
    max_depth::Int = 10,
    min_node_size::Int = 5,
    mtry::Union{Nothing, Int} = nothing,
    optimization_method::Function = estimate_parameters_mom
)
```

The `q_threshold` parameter controls how many candidate split thresholds are considered per feature.

By default, `q_threshold` is set to a very large number (`500000000`), which causes the algorithm to use **all unique values** of each feature as split candidates.

This approach can be inefficient, especially when working with continuous features that have many unique values.

To improve efficiency and potentially improve generalization, you can set `q_threshold` to a smaller number.

For example, setting `q_threshold = 4` will use only the **0.25, 0.50, and 0.75 quantiles** of each feature as candidate split points.

This adjustment is particularly useful when dealing with **high-cardinality continuous covariates**.

## Example in Julia

To see an example of how to use `Dirichlet_RF` in Julia, refer to the `Dirichlet_RF_Example.jl` file located in the `examples` directory.
If you want to run the code in a distributed manner, check: `examples/Dirichlet_RF_Example_distributed.jl`.
By default, the script adds worker processes with: `addprocs(Sys.CPU_THREADS - 1)`. 
Here, `Sys.CPU_THREADS detects` the total number of CPU threads on your system.

For example, if your system has 10 threads and you want to use only 5, change: `addprocs(Sys.CPU_THREADS - 1)` to: `addprocs(Sys.CPU_THREADS - 5)`. This lets you control how many CPU threads are allocated for parallel execution.

## Using from R

You can also use these Julia functions from within R. To get started, install Julia and set up the `JuliaCall` package in R:

```r
if (!require("JuliaCall")) install.packages("JuliaCall")
library(JuliaCall)
install_julia()
```
You can see an example in the examples folder: `call_to_R.R`.

This part is not yet complete: The main function for calling the Julia code from R is defined in the `dirichlet_forest_wrapper.R` file located in the src folder.
To see how to use it in practice, refer to the `example_usage_dirichlet_forest.R` file in the examples folder.

## Code Related to the Paper: *Dirichlet Random Forest for Predicting Compositional Data*

For the simple and complex examples discussed in the paper, please refer to the following scripts:  
- `example_simple_structure.R`  
- `example_complex_structure.R`  
- `comparision_models`  
- `rf_for_call_to_R.jl`

