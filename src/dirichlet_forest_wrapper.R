# Dirichlet Random Forest R Wrapper
# Professional interface to Julia implementation
# Author: Your Name
# Date: 2025

#' Initialize Dirichlet Forest Julia Environment
#' 
#' Sets up the Julia environment and loads all required packages and functions
#' 
#' @param julia_file_path Path to the main Julia file (dirichlet_forest_ml.jl)
#' @param verbose Logical, whether to print setup messages
#' @return TRUE if successful, FALSE otherwise
#' 
#' @export
initialize_dirichlet_forest <- function(julia_file_path = NULL, verbose = TRUE) {
  
  # Check if JuliaCall is available
  if (!requireNamespace("JuliaCall", quietly = TRUE)) {
    stop("JuliaCall package is required. Please install it using: install.packages('JuliaCall')")
  }
  
  # Load JuliaCall
  suppressPackageStartupMessages(library(JuliaCall, quietly = TRUE))
  
  # Initialize Julia
  tryCatch({
    if (verbose) cat("Initializing Julia environment...\n")
    julia_setup()
  }, error = function(e) {
    stop("Failed to initialize Julia. Please ensure Julia is properly installed. Error: ", e$message)
  })
  
  # Define required Julia packages
  required_packages <- c("Random", "Distributions", "CSV", "DataFrames", 
                         "StatsBase", "Statistics", "SpecialFunctions", 
                         "Optim", "NLopt", "BenchmarkTools", "LinearAlgebra", 
                         "ForwardDiff")
  
  # Install and load Julia packages
  if (verbose) cat("Checking and installing required Julia packages...\n")
  for (pkg in required_packages) {
    tryCatch({
      julia_install_package_if_needed(pkg)
    }, error = function(e) {
      warning("Failed to install Julia package: ", pkg, ". Error: ", e$message)
    })
  }
  
  # Load Julia packages
  tryCatch({
    julia_command("using Random, Distributions, CSV, DataFrames, StatsBase, Statistics, SpecialFunctions, Optim, NLopt, BenchmarkTools, LinearAlgebra, ForwardDiff")
  }, error = function(e) {
    stop("Failed to load Julia packages. Error: ", e$message)
  })
  
  # Source Julia files if path provided
  if (!is.null(julia_file_path)) {
    if (!file.exists(julia_file_path)) {
      stop("Julia file not found at: ", julia_file_path)
    }
    
    tryCatch({
      if (verbose) cat("Loading Dirichlet Forest Julia implementation...\n")
      julia_source(julia_file_path)
    }, error = function(e) {
      stop("Failed to source Julia file. Error: ", e$message)
    })
  }
  
  if (verbose) cat("Julia environment initialized successfully!\n")
  return(TRUE)
}

#' Fit Dirichlet Random Forest
#' 
#' Trains a Dirichlet Random Forest model using Julia implementation
#' 
#' @param X_train Training feature matrix (n_samples x n_features)
#' @param Y_train Training response matrix (n_samples x n_categories), should be compositional data
#' @param n_trees Number of trees in the forest (default: 500)
#' @param q_threshold Quantile threshold for splits (default: 500000000)
#' @param max_depth Maximum depth of trees (default: 200000)
#' @param min_node_size Minimum samples required to split a node (default: 5)
#' @param mtry Number of variables to try at each split (default: sqrt(n_features))
#' @param estimation_method Parameter estimation method: "mom", "mle_newton", "mle_bfgs", "mle_nelder_mead", "mle_bobyqa", "mle_gbdt" (default: "mle_newton")
#' @param verbose Logical, whether to print progress messages
#' 
#' @return A list containing the fitted forest object and other components
#' 
#' @export
fit_dirichlet_forest <- function(X_train, Y_train, 
                                  n_trees = 500,
                                  q_threshold = 500000000,
                                  max_depth = 200000,
                                  min_node_size = 5,
                                  mtry = NULL,
                                  estimation_method = "mle_newton",
                                  verbose = TRUE) {
  
  # Input validation
  if (!is.matrix(X_train)) X_train <- as.matrix(X_train)
  if (!is.matrix(Y_train)) Y_train <- as.matrix(Y_train)
  
  # Set default mtry if not provided
  if (is.null(mtry)) {
    mtry <- max(1, floor(sqrt(ncol(X_train))))
  }
  
  # Validate estimation method
  valid_methods <- c("mom", "mle_newton", "mle_bfgs", "mle_nelder_mead", "mle_bobyqa", "mle_gbdt")
  if (!estimation_method %in% valid_methods) {
    stop("estimation_method must be one of: ", paste(valid_methods, collapse = ", "))
  }
  
  # Transfer data to Julia
  if (verbose) cat("Transferring data to Julia...\n")
  julia_assign("x_train", X_train)
  julia_assign("y_train", Y_train)
  
  # Convert data types in Julia
  julia_eval("x_train = convert(Matrix{Float64}, x_train)")
  julia_eval("y_train = convert(Matrix{Float64}, y_train)")
  
  # Set parameters in Julia
  julia_assign("n_trees", as.integer(n_trees))
  julia_assign("q_threshold", as.integer(q_threshold))
  julia_assign("max_depth", as.integer(max_depth))
  julia_assign("min_node_size", as.integer(min_node_size))
  julia_assign("mtry", as.integer(mtry))
  
  # Initialize forest
  julia_eval("forest = DirichletForest(n_trees)")
  
  # Choose estimation function
  estimation_func <- paste0("estimate_parameters_", estimation_method)
  
  # Train the forest
  if (verbose) cat("Training Dirichlet Random Forest with", n_trees, "trees...\n")
  julia_command <- paste0("fit_dirichlet_forest!(forest, x_train, y_train, q_threshold, max_depth, min_node_size, mtry, ", estimation_func, ")")
  julia_eval(julia_command)
  
  # Store the forest object in Julia global environment
  julia_eval("global fitted_forest = forest")
  
  if (verbose) cat("Dirichlet Random Forest training completed successfully!\n")
  
  # Return model parameters and info
  result <- list(
    fitted = TRUE,
    parameters = list(
      n_trees = n_trees,
      q_threshold = q_threshold,
      max_depth = max_depth,
      min_node_size = min_node_size,
      mtry = mtry,
      estimation_method = estimation_method
    ),
    training_data_shape = dim(X_train),
    response_data_shape = dim(Y_train)
  )
  
  class(result) <- "DirichletForest"
  return(result)
}

#' Predict using Dirichlet Random Forest
#' 
#' Make predictions using a fitted Dirichlet Random Forest model
#' 
#' @param forest_model A fitted DirichletForest object from fit_dirichlet_forest()
#' @param X_new New data matrix for prediction (n_samples x n_features)
#' @param verbose Logical, whether to print progress messages
#' 
#' @return Matrix of predictions (n_samples x n_categories)
#' 
#' @export
predict_dirichlet_forest <- function(forest_model, X_new, verbose = TRUE) {
  
  # Check if model is fitted
  if (!inherits(forest_model, "DirichletForest") || !forest_model$fitted) {
    stop("Model must be a fitted DirichletForest object")
  }
  
  # Input validation
  if (!is.matrix(X_new)) X_new <- as.matrix(X_new)
  
  # Transfer new data to Julia
  if (verbose) cat("Making predictions...\n")
  julia_assign("x_new", X_new)
  julia_eval("x_new = convert(Matrix{Float64}, x_new)")
  
  # Make predictions
  julia_eval("predictions = predict_dirichlet_forest(fitted_forest, x_new)")
  predictions <- julia_eval("predictions")
  predictions <- as.matrix(predictions)
  
  if (verbose) cat("Predictions completed successfully!\n")
  
  return(predictions)
}

#' Get Feature Importance from Dirichlet Random Forest
#' 
#' Extract feature importance measures from a fitted model
#' 
#' @param forest_model A fitted DirichletForest object
#' @param feature_names Optional vector of feature names
#' 
#' @return Data frame with feature importance measures
#' 
#' @export
get_feature_importance <- function(forest_model, feature_names = NULL) {
  
  # Check if model is fitted
  if (!inherits(forest_model, "DirichletForest") || !forest_model$fitted) {
    stop("Model must be a fitted DirichletForest object")
  }
  
  # Get importance measures from Julia
  importance_values <- julia_eval("fitted_forest.importance")
  importance_freq <- julia_eval("fitted_forest.importancef")
  
  # Create feature names if not provided
  if (is.null(feature_names)) {
    n_features <- length(importance_values)
    feature_names <- paste0("X", 1:n_features)
  }
  
  # Create feature importance dataframe
  feature_importance <- data.frame(
    feature = feature_names,
    importance = as.numeric(importance_values),
    frequency = as.numeric(importance_freq),
    stringsAsFactors = FALSE
  )
  
  # Sort by importance
  feature_importance <- feature_importance[order(feature_importance$importance, decreasing = TRUE), ]
  
  return(feature_importance)
}

#' Fit Dirichlet Random Forest with Parallel Processing
#' 
#' Trains a Dirichlet Random Forest model using parallel Julia implementation
#' 
#' @param X_train Training feature matrix (n_samples x n_features)
#' @param Y_train Training response matrix (n_samples x n_categories)
#' @param n_trees Number of trees in the forest (default: 500)
#' @param q_threshold Quantile threshold for splits (default: 500000000)
#' @param max_depth Maximum depth of trees (default: 200000)
#' @param min_node_size Minimum samples required to split a node (default: 5)
#' @param mtry Number of variables to try at each split (default: sqrt(n_features))
#' @param estimation_method Parameter estimation method (default: "mle_newton")
#' @param julia_parallel_file Path to parallel Julia file (default: NULL)
#' @param verbose Logical, whether to print progress messages
#' 
#' @return A list containing the fitted forest object
#' 
#' @export
fit_dirichlet_forest_parallel <- function(X_train, Y_train,
                                           n_trees = 500,
                                           q_threshold = 500000000,
                                           max_depth = 200000,
                                           min_node_size = 5,
                                           mtry = NULL,
                                           estimation_method = "mle_newton",
                                           julia_parallel_file = NULL,
                                           verbose = TRUE) {
  
  # Load parallel implementation if specified
  if (!is.null(julia_parallel_file)) {
    if (!file.exists(julia_parallel_file)) {
      stop("Parallel Julia file not found at: ", julia_parallel_file)
    }
    julia_source(julia_parallel_file)
  }
  
  # Input validation
  if (!is.matrix(X_train)) X_train <- as.matrix(X_train)
  if (!is.matrix(Y_train)) Y_train <- as.matrix(Y_train)
  
  # Set default mtry if not provided
  if (is.null(mtry)) {
    mtry <- max(1, floor(sqrt(ncol(X_train))))
  }
  
  # Transfer data to Julia
  if (verbose) cat("Transferring data to Julia for parallel processing...\n")
  julia_assign("x_train", X_train)
  julia_assign("y_train", Y_train)
  
  # Convert data types in Julia
  julia_eval("x_train = convert(Matrix{Float64}, x_train)")
  julia_eval("y_train = convert(Matrix{Float64}, y_train)")
  
  # Set parameters
  julia_assign("n_trees", as.integer(n_trees))
  julia_assign("q_threshold", as.integer(q_threshold))
  julia_assign("max_depth", as.integer(max_depth))
  julia_assign("min_node_size", as.integer(min_node_size))
  julia_assign("mtry", as.integer(mtry))
  
  # Initialize forest
  julia_eval("forest = DirichletForest(n_trees)")
  
  # Choose estimation function
  estimation_func <- paste0("estimate_parameters_", estimation_method)
  
  # Train the forest with parallel processing
  if (verbose) cat("Training Dirichlet Random Forest with parallel processing...\n")
  julia_command <- paste0("fit_dirichlet_forest_parallel!(forest, x_train, y_train, q_threshold, max_depth, min_node_size, mtry, ", estimation_func, ")")
  julia_eval(julia_command)
  
  # Store the forest object
  julia_eval("global fitted_forest = forest")
  
  if (verbose) cat("Parallel Dirichlet Random Forest training completed successfully!\n")
  
  # Return model info
  result <- list(
    fitted = TRUE,
    parallel = TRUE,
    parameters = list(
      n_trees = n_trees,
      q_threshold = q_threshold,
      max_depth = max_depth,
      min_node_size = min_node_size,
      mtry = mtry,
      estimation_method = estimation_method
    ),
    training_data_shape = dim(X_train),
    response_data_shape = dim(Y_train)
  )
  
  class(result) <- "DirichletForest"
  return(result)
}

#' Clean up Julia Environment
#' 
#' Removes workers and cleans up Julia environment
#' 
#' @export
cleanup_julia_environment <- function() {
  tryCatch({
    julia_eval("if @isdefined(workers); rmprocs(workers()); end")
    cat("Julia environment cleaned up successfully.\n")
  }, error = function(e) {
    warning("Error during cleanup: ", e$message)
  })
}

#' Print method for DirichletForest objects
#' 
#' @param x A DirichletForest object
#' @param ... Additional arguments (ignored)
#' 
#' @export
print.DirichletForest <- function(x, ...) {
  cat("Dirichlet Random Forest Model\n")
  cat("=============================\n")
  cat("Fitted:", x$fitted, "\n")
  if ("parallel" %in% names(x)) {
    cat("Parallel processing:", x$parallel, "\n")
  }
  cat("\nParameters:\n")
  for (param in names(x$parameters)) {
    cat("  ", param, ":", x$parameters[[param]], "\n")
  }
  cat("\nTraining data shape:", paste(x$training_data_shape, collapse = " x "), "\n")
  cat("Response data shape:", paste(x$response_data_shape, collapse = " x "), "\n")
}

#' Summary method for DirichletForest objects
#' 
#' @param object A DirichletForest object
#' @param ... Additional arguments (ignored)
#' 
#' @export
summary.DirichletForest <- function(object, ...) {
  print(object)
  if (object$fitted) {
    cat("\nFeature Importance (Top 10):\n")
    importance <- get_feature_importance(object)
    print(head(importance, 10))
  }
}