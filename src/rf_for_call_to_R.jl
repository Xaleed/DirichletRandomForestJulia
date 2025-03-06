# Save this as simulation_predictor.jl
using Random, Distributions, CSV, DataFrames, StatsBase, Statistics
include("dirichlet_forest_ml.jl")
include("dirichlet_forest_ml_distributed.jl")

#include("dirichlet_forest_ml_parallel.jl")


function train_and_predict(X_train, Y_train, X_test; 
                         n_trees=500, 
                         q_threshold=500000000,
                         max_depth=200000,
                         min_node_size=5,
                         mtry = 15)
    
    # Initialize forest
    forest = DirichletForest(n_trees)
    
    # Train forest using MLE-Newton method
    println("Training forest...")
    fit_dirichlet_forest!(
        forest, 
        X_train, 
        Y_train, 
        q_threshold, 
        max_depth, 
        min_node_size,
        mtry,
        estimate_parameters_mle_newton
    )
    
    # Make predictions
    println("Making predictions...")
    pred_test = predict_dirichlet_forest(forest, X_test)
    pred_train = predict_dirichlet_forest(forest, X_train)
    
    return pred_train, pred_test, forest.importance, forest.importancef
end

function train_and_predict(X_train, Y_train, X_test; 
                         n_trees=500, 
                         q_threshold=500000000,
                         max_depth=200000,
                         min_node_size=5,
                         mtry = 15)
    
    # Initialize forest
    forest = DirichletForest(n_trees)
    
    # Train forest using MLE-Newton method
    println("Training forest...")
    fit_dirichlet_forest!(
        forest, 
        X_train, 
        Y_train, 
        q_threshold, 
        max_depth, 
        min_node_size,
        mtry,
        estimate_parameters_mle_newton
    )
    
    # Make predictions
    println("Making predictions...")
    pred_test = predict_dirichlet_forest(forest, X_test)
    pred_train = predict_dirichlet_forest(forest, X_train)
    
    return pred_train, pred_test, forest.importance, forest.importancef
end

function train_and_predict_parallel(X_train, Y_train, X_test; 
                         n_trees=500, 
                         q_threshold=500000000,
                         max_depth=200000,
                         min_node_size=5,
                         mtry = 15)
    
    # Initialize forest
    forest = DirichletForest(n_trees)
    
    # Train forest using MLE-Newton method
    println("Training forest...")
    fit_dirichlet_forest_parallel!(
        forest, 
        X_train, 
        Y_train, 
        q_threshold, 
        max_depth, 
        min_node_size,
        mtry,
        estimate_parameters_mle_newton
    )
    
    # Make predictions
    println("Making predictions...")
    pred_test = predict_dirichlet_forest(forest, X_test)
    pred_train = predict_dirichlet_forest(forest, X_train)
    
    return pred_train, pred_test, forest.importance, forest.importancef
end

# Function to process data from R
function process_matrix_data(x_train, y_train, x_test)
    x_train = convert(Matrix{Float64}, x_train)
    y_train = convert(Matrix{Float64}, y_train)
    x_test = convert(Matrix{Float64}, x_test)
    
    return x_train, y_train, x_test
end