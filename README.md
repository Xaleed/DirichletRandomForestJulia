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
│   ├── DirichletRandomForest.jl    # Main module file
│   ├── MLE_vs_MoM.jl              # Parameter estimation methods
│   ├── dirichlet_forest.jl         # Main implementation
│   └── evaluation.jl               # Evaluation metrics
│
├── test/
│   ├── runtests.jl                 # Main test file
│   ├── test_mle_mom.jl             # Tests for parameter estimation
│   ├── test_forest.jl              # Tests for main implementation
│   └── test_evaluation.jl          # Tests for evaluation metrics
│
├── docs/
│   ├── src/                        # Documentation source files
│   └── make.jl                     # Documentation generation script
│
├── Project.toml                    # Project dependencies and metadata
├── README.md                       # This file
└── LICENSE                         # License information
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

## Usage

```julia
using DirichletRandomForest

# Example code will go here
```

## Features

- Implementation of Random Forests for Dirichlet distributions
- Multiple parameter estimation methods (MLE and Method of Moments)
- Comprehensive evaluation metrics

## Testing

To run the tests, execute:

```julia
using Pkg
Pkg.test("DirichletRandomForest")
```

## Documentation

Documentation is available at [link to documentation].

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [LICENSE] - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```
Citation information will go here
```

## Authors

Khaled Masoumifar, Stephan van der Westhuizen