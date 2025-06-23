# DirichletRandomForest

A Julia package for implementing Random Forests with Dirichlet distributions.

## Installation

To install this package, use Julia's package manager:

```julia
using Pkg
Pkg.add("DirichletRandomForest")
```

## Project Structure

```
DirichletRandomForest/
├── src/
│   ├── julia/
│   │   ├── DirichletRandomForest.jl       # Main module
│   │   ├── dirichlet_forest.jl           # Core algorithm
│   │   ├── parameter_estimation.jl       # MLE/MoM implementations
│   │   └── r_interface.jl                # Julia functions called from R
│   │
│   └── r/
│       ├── model_comparison.R            # compare_models function
│       ├── model_utils.R                 # Evaluation metrics
│       └── dirichlet_regression.R        # Dirichlet regression helpers
├── examples/
│   ├── real_data_analysis.R              # Cleaned version of example_real_data.R
│   └── simulation_study.R                # Cleaned version of example_simple_structure.R
├── test/
│   ├── julia/
│   │   └── test_core.jl                  # Unit tests for Julia
│   └── r/
│       └── test_models.R                 # Model comparison tests
├── docs/
│   ├── make.jl                           # Documentation builder
│   └── src/                              # Documentation source files
├── Project.toml                          # Julia dependencies
├── DESCRIPTION                           # R package metadata
├── NAMESPACE                             # R namespace
└── README.md
```

## Dependencies

- Julia 1.6 or higher
- Distributions.jl (0.25)
- DataFrames.jl (1.3)
- ForwardDiff.jl (0.10)
- LinearAlgebra.jl
- NLopt.jl (0.6)
- Optim.jl (1.7)
- Random.jl
- SpecialFunctions.jl (2.1)
- Statistics.jl


## Authors

Khaled Masoumifar, Stephan van der Westhuizen
