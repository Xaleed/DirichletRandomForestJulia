# Example Usage of Dirichlet Random Forest R Wrapper
# This file demonstrates how to use the Dirichlet Forest functions

# Clear workspace
rm(list = ls())

# Load required libraries
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}
if (!requireNamespace("reshape2", quietly = TRUE)) {
  install.packages("reshape2")
}
library(ggplot2)
library(reshape2)

# Source the wrapper functions
source("C:/Users/29827094/Documents/GitHub/DirichletRandomForestJulia/src/dirichlet_forest_wrapper.R")
source("C:/Users/29827094/Documents/GitHub/DirichletRandomForestJulia/src/evaluation_utils.R")

# =============================================================================
# STEP 1: SETUP AND INITIALIZATION
# =============================================================================

# Set paths to your Julia files
julia_main_file <- "C:/Users/29827094/Documents/GitHub/DirichletRandomForestJulia/src/dirichlet_forest_ml.jl"
julia_parallel_file <- "C:/Users/29827094/Documents/GitHub/DirichletRandomForestJulia/src/dirichlet_forest_ml_distributed.jl"

# Initialize Julia environment
cat("Initializing Dirichlet Forest environment...\n")
initialize_dirichlet_forest(julia_file_path = julia_main_file, verbose = TRUE)

# =============================================================================
# STEP 2: GENERATE EXAMPLE DATA
# =============================================================================

#' Generate Compositional Data for Testing
#' 
#' Creates synthetic compositional data with known relationships
generate_compositional_data <- function(n_samples = 500, n_features = 10, n_categories = 3, seed = 123) {
  set.seed(seed)
  
  # Generate feature matrix
  X <- matrix(0, nrow = n_samples, ncol = n_features)
  colnames(X) <- paste0("X", 1:n_features)
  
  # Mix of continuous and categorical features
  for (i in 1:n_features) {
    if (i <= 3) {
      # Continuous features
      X[, i] <- runif(n_samples, -2, 2)
    } else {
      # Categorical features (encoded as numeric)
      X[, i] <- sample(1:5, n_samples, replace = TRUE)
    }
  }
  
  # Generate compositional response with complex relationships
  alpha <- matrix(0, nrow = n_samples, ncol = n_categories)
  
  # Create non-linear relationships
  alpha[, 1] <- exp(1 + 0.5 * X[, 1] + 0.3 * sin(X[, 2]) + 0.2 * X[, 4])
  alpha[, 2] <- exp(0.8 + 0.7 * X[, 2]^2 + 0.4 * X[, 3] + 0.1 * X[, 5])  
  alpha[, 3] <- exp(1.2 + 0.6 * cos(X[, 1]) + 0.5 * X[, 3] * X[, 2] + 0.3 * X[, 6])
  
  # Add more categories if needed
  if (n_categories > 3) {
    for (k in 4:n_categories) {
      alpha[, k] <- exp(runif(1, 0.5, 1.5) + 
                          0.4 * X[, min(k, n_features)] + 
                          0.2 * X[, min(k+1, n_features)])
    }
  }
  
  # Ensure minimum alpha values
  alpha <- pmax(alpha, 0.1)
  
  # Generate Dirichlet-like compositional data
  Y <- matrix(0, nrow = n_samples, ncol = n_categories)
  for (i in 1:n_samples) {
    # Generate gamma random variables and normalize
    gammas <- rgamma(n_categories, shape = alpha[i, ], rate = 1)
    Y[i, ] <- gammas / sum(gammas)
  }
  
  # Ensure valid compositional data
  Y <- pmax(Y, 1e-6)  # Avoid zeros
  Y <- Y / rowSums(Y)  # Ensure sum to 1
  
  return(list(X = X, Y = Y, alpha = alpha))
}

#' Split data into training and testing sets
split_data <- function(X, Y, train_ratio = 0.8, seed = 123) {
  set.seed(seed)
  n_samples <- nrow(X)
  n_train <- floor(train_ratio * n_samples)
  
  train_indices <- sample(1:n_samples, n_train, replace = FALSE)
  test_indices <- setdiff(1:n_samples, train_indices)
  
  return(list(
    X_train = X[train_indices, , drop = FALSE],
    Y_train = Y[train_indices, , drop = FALSE],
    X_test = X[test_indices, , drop = FALSE],
    Y_test = Y[test_indices, , drop = FALSE],
    train_indices = train_indices,
    test_indices = test_indices
  ))
}

# Generate example data
cat("Generating compositional data...\n")
data_config <- list(
  n_samples = 100,
  n_features = 8,
  n_categories = 3,
  seed = 42
)

sim_data <- generate_compositional_data(
  n_samples = data_config$n_samples,
  n_features = data_config$n_features, 
  n_categories = data_config$n_categories,
  seed = data_config$seed
)

# Split the data
split_config <- list(train_ratio = 0.7, seed = 42)
data_split <- split_data(sim_data$X, sim_data$Y, 
                         train_ratio = split_config$train_ratio, 
                         seed = split_config$seed)

cat(sprintf("Data split: %d training, %d testing samples\n", 
            nrow(data_split$X_train), nrow(data_split$X_test)))

# =============================================================================
# STEP 3: BASIC MODEL FITTING
# =============================================================================

# Configure model parameters
model_params <- list(
  n_trees = 100,
  max_depth = 10,
  min_node_size = 5,
  mtry = NULL,  # Will default to sqrt(n_features)
  q_threshold = 50,
  estimation_method = "mle_newton"
)

cat("Training Dirichlet Random Forest...\n")
cat(sprintf("Parameters: %d trees, max_depth=%d, min_node_size=%d\n", 
            model_params$n_trees, model_params$max_depth, model_params$min_node_size))

# Fit the model
drf_model <- fit_dirichlet_forest(
  X = data_split$X_train,
  Y = data_split$Y_train,
  n_trees = model_params$n_trees,
  max_depth = model_params$max_depth,
  min_node_size = model_params$min_node_size,
  mtry = model_params$mtry,
  q_threshold = model_params$q_threshold,
  estimation_method = model_params$estimation_method,
  verbose = TRUE
)

# Print model summary
print(drf_model)
summary(drf_model)

# =============================================================================
# STEP 4: MAKE PREDICTIONS
# =============================================================================

cat("Making predictions...\n")

# Predict on training data
train_predictions <- predict_dirichlet_forest(drf_model, data_split$X_train)

# Predict on test data  
test_predictions <- predict_dirichlet_forest(drf_model, data_split$X_test)

cat(sprintf("Training predictions shape: %d x %d\n", 
            nrow(train_predictions), ncol(train_predictions)))
cat(sprintf("Test predictions shape: %d x %d\n", 
            nrow(test_predictions), ncol(test_predictions)))

# =============================================================================
# STEP 5: EVALUATE PERFORMANCE
# =============================================================================

cat("Evaluating model performance...\n")

# Evaluate training performance
train_metrics <- evaluate_performance(data_split$Y_train, train_predictions)
cat("Training Performance:\n")
print(train_metrics)

# Evaluate test performance
test_metrics <- evaluate_performance(data_split$Y_test, test_predictions)
cat("\nTest Performance:\n")
print(test_metrics)

# Create performance comparison
performance_comparison <- data.frame(
  Metric = c("Aitchison Distance", "Compositional R²", "Mean Euclidean", "RMSE"),
  Training = c(train_metrics$aitchison_distance, train_metrics$compositional_r2, 
               train_metrics$mean_euclidean, train_metrics$rmse),
  Test = c(test_metrics$aitchison_distance, test_metrics$compositional_r2,
           test_metrics$mean_euclidean, test_metrics$rmse)
)

print("\nPerformance Comparison:")
print(performance_comparison)

# =============================================================================
# STEP 6: FEATURE IMPORTANCE ANALYSIS
# =============================================================================

cat("Analyzing feature importance...\n")

# Get feature importance
importance_scores <- get_feature_importance(drf_model)
print("Feature Importance:")
print(importance_scores)

# Create feature importance plot
importance_plot <- ggplot(importance_scores, aes(x = reorder(feature, importance), y = importance)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
  coord_flip() +
  theme_minimal() +
  labs(title = "Dirichlet Random Forest - Feature Importance",
       x = "Features", 
       y = "Importance Score") +
  theme(plot.title = element_text(hjust = 0.5))

print(importance_plot)

# =============================================================================
# STEP 7: CROSS-VALIDATION
# =============================================================================

cat("Running cross-validation...\n")

# Configure cross-validation
cv_config <- list(
  k_folds = 5,
  n_trees = 50,  # Fewer trees for faster CV
  max_depth = 8,
  min_node_size = 5,
  seed = 123
)

# Run cross-validation
cv_results <- cv_dirichlet_forest(
  X = data_split$X_train,
  Y = data_split$Y_train,
  k_folds = cv_config$k_folds,
  n_trees = cv_config$n_trees,
  max_depth = cv_config$max_depth,
  min_node_size = cv_config$min_node_size,
  estimation_method = "estimate_parameters_mle_newton",
  seed = cv_config$seed,
  verbose = TRUE
)

cat("Cross-Validation Results:\n")
print(cv_results$summary)

# =============================================================================
# STEP 8: MODEL COMPARISON
# =============================================================================

cat("Comparing different estimation methods...\n")

# Compare MLE Newton vs Method of Moments
estimation_methods <- c("mle_newton", "mom")
comparison_results <- list()

for (method in estimation_methods) {
  cat(sprintf("Training with %s estimation...\n", method))
  
  model <- fit_dirichlet_forest(
    X = data_split$X_train,
    Y = data_split$Y_train,
    n_trees = 50,  # Fewer trees for comparison
    max_depth = 8,
    min_node_size = 5,
    estimation_method = method,
    verbose = FALSE
  )
  
  # Make predictions
  train_pred <- predict_dirichlet_forest(model, data_split$X_train)
  test_pred <- predict_dirichlet_forest(model, data_split$X_test)
  
  # Evaluate performance
  train_perf <- evaluate_performance(data_split$Y_train, train_pred)
  test_perf <- evaluate_performance(data_split$Y_test, test_pred)
  
  comparison_results[[method]] <- list(
    train = train_perf,
    test = test_perf,
    model = model
  )
}

# Create comparison summary
method_comparison <- data.frame(
  Method = rep(estimation_methods, each = 4),
  Dataset = rep(c("Train", "Test"), each = 2, times = 2),
  Metric = rep(c("Aitchison Distance", "RMSE"), times = 4),
  Value = c(
    comparison_results$mle_newton$train$aitchison_distance,
    comparison_results$mle_newton$train$rmse,
    comparison_results$mle_newton$test$aitchison_distance,
    comparison_results$mle_newton$test$rmse,
    comparison_results$mom$train$aitchison_distance,
    comparison_results$mom$train$rmse,
    comparison_results$mom$test$aitchison_distance,
    comparison_results$mom$test$rmse
  )
)

print("Method Comparison:")
print(method_comparison)

# Plot method comparison
comparison_plot <- ggplot(method_comparison, aes(x = Method, y = Value, fill = Dataset)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.7) +
  facet_wrap(~Metric, scales = "free_y") +
  theme_minimal() +
  labs(title = "Estimation Method Comparison",
       x = "Estimation Method",
       y = "Performance Value") +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_fill_brewer(palette = "Set2")

print(comparison_plot)

# =============================================================================
# STEP 9: PARALLEL PROCESSING EXAMPLE
# =============================================================================

cat("Demonstrating parallel processing...\n")

# Note: This requires the parallel Julia file to be loaded
# Uncomment the following lines if you want to test parallel processing

# tryCatch({
#   # Initialize parallel environment
#   initialize_dirichlet_forest(julia_file_path = julia_parallel_file, verbose = TRUE)
#   
#   # Fit model using parallel processing
#   parallel_model <- fit_dirichlet_forest_parallel(
#     X = data_split$X_train,
#     Y = data_split$Y_train,
#     n_trees = 100,
#     max_depth = 10,
#     estimation_method = "mle_newton",
#     verbose = TRUE
#   )
#   
#   cat("Parallel model training completed successfully!\n")
#   print(parallel_model)
#   
# }, error = function(e) {
#   cat("Parallel processing not available or failed:", e$message, "\n")
# })

# =============================================================================
# STEP 10: VISUALIZATION AND REPORTING
# =============================================================================

cat("Creating visualizations...\n")

# Plot actual vs predicted for test set (first component)
prediction_df <- data.frame(
  Actual = data_split$Y_test[, 1],
  Predicted = test_predictions[, 1],
  Component = "Component 1"
)

# Add other components
for (i in 2:ncol(data_split$Y_test)) {
  temp_df <- data.frame(
    Actual = data_split$Y_test[, i],
    Predicted = test_predictions[, i],
    Component = paste("Component", i)
  )
  prediction_df <- rbind(prediction_df, temp_df)
}

# Actual vs Predicted plot
pred_plot <- ggplot(prediction_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  facet_wrap(~Component, scales = "free") +
  theme_minimal() +
  labs(title = "Actual vs Predicted Values",
       x = "Actual Values",
       y = "Predicted Values") +
  theme(plot.title = element_text(hjust = 0.5))

print(pred_plot)

# Performance metrics summary
final_summary <- data.frame(
  Dataset = c("Training", "Test"),
  `Aitchison Distance` = c(train_metrics$aitchison_distance, test_metrics$aitchison_distance),
  `Compositional R²` = c(train_metrics$compositional_r2, test_metrics$compositional_r2),
  `Mean Euclidean` = c(train_metrics$mean_euclidean, test_metrics$mean_euclidean),
  `RMSE` = c(train_metrics$rmse, test_metrics$rmse),
  check.names = FALSE
)

cat("\nFinal Performance Summary:\n")
print(final_summary)

# =============================================================================
# STEP 11: CLEANUP
# =============================================================================

cat("Cleaning up Julia environment...\n")
cleanup_julia_environment()

cat("\nDirichlet Random Forest example completed successfully!\n")
cat("Results summary:\n")
cat(sprintf("- Model trained with %d trees\n", model_params$n_trees))
cat(sprintf("- Training samples: %d\n", nrow(data_split$X_train)))
cat(sprintf("- Test samples: %d\n", nrow(data_split$X_test))) 
cat(sprintf("- Features: %d\n", ncol(data_split$X_train)))
cat(sprintf("- Categories: %d\n", ncol(data_split$Y_train)))
cat(sprintf("- Test Aitchison Distance: %.4f\n", test_metrics$aitchison_distance))
cat(sprintf("- Test Compositional R²: %.4f\n", test_metrics$compositional_r2))

# =============================================================================
# ADDITIONAL HELPER FUNCTIONS
# =============================================================================

#' Save model results to file
save_results <- function(model, predictions, metrics, file_prefix = "drf_results") {
  # Create results directory if it doesn't exist
  if (!dir.exists("results")) {
    dir.create("results")
  }
  
  # Save model summary
  sink(file.path("results", paste0(file_prefix, "_model_summary.txt")))
  print(model)
  summary(model)
  sink()
  
  # Save predictions
  write.csv(predictions, file.path("results", paste0(file_prefix, "_predictions.csv")), row.names = FALSE)
  
  # Save metrics
  write.csv(metrics, file.path("results", paste0(file_prefix, "_metrics.csv")), row.names = FALSE)
  
  cat(sprintf("Results saved to 'results' directory with prefix '%s'\n", file_prefix))
}

# Uncomment to save results
# save_results(drf_model, test_predictions, test_metrics, "example_drf")