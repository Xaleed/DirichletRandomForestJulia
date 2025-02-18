library(MCMCpack)

library(parallel)

library(doParallel)
library(DirichletReg)
library(compositions)
library(randomForest)




# Then load them
library(xgboost)
library(keras)
library(magrittr)

####################
#keras::install_keras()




# Calculate geometric mean
geometric_mean <- function(x) {
  exp(mean(log(x)))
}
################################

# Evaluate performance
evaluate_performance <- function(Y_true, Y_pred) {
  n_samples <- nrow(Y_true)
  
  aitchison_dist <- mean(sapply(1:n_samples, function(i) {
    y_true <- as.numeric(Y_true[i,])
    y_pred <- as.numeric(Y_pred[i,])
    
    y_true <- pmax(y_true, .Machine$double.eps)
    y_pred <- pmax(y_pred, .Machine$double.eps)
    
    gm_true <- geometric_mean(y_true)
    gm_pred <- geometric_mean(y_pred)
    
    sqrt(sum((log(y_true/gm_true) - log(y_pred/gm_pred))^2))
  }))
  
  total_var <- sum(sapply(1:n_samples, function(i) {
    y_true <- as.numeric(Y_true[i,])
    y_true <- pmax(y_true, .Machine$double.eps)
    gm_true <- geometric_mean(y_true)
    sum((log(y_true/gm_true))^2)
  }))
  
  residual_var <- sum(sapply(1:n_samples, function(i) {
    y_true <- as.numeric(Y_true[i,])
    y_pred <- as.numeric(Y_pred[i,])
    
    y_true <- pmax(y_true, .Machine$double.eps)
    y_pred <- pmax(y_pred, .Machine$double.eps)
    
    gm_true <- geometric_mean(y_true)
    gm_pred <- geometric_mean(y_pred)
    
    sum((log(y_true/gm_true) - log(y_pred/gm_pred))^2)
  }))
  
  comp_r2 <- 1 - residual_var/total_var
  rmse <- sqrt(colMeans((Y_true - Y_pred)^2))
  
  return(list(
    aitchison_distance = aitchison_dist,
    compositional_r2 = comp_r2,
    component_rmse = rmse,
    mean_rmse = mean(rmse)
  ))
}
############################

library(MASS)  # For rdirichlet function
set.seed(48)
n_samples <- 500
n_features <- 10
n_categories <- 3
# Generate feature matrix
X <- matrix(runif(n_samples * n_features), n_samples, n_features)
colnames(X) <- paste0("X", 1:n_features)
# Create alpha parameters that favor DRF-MLE over ILR-RF
alpha <- matrix(0, nrow=n_samples, ncol=n_categories)
# Key modifications to favor DRF-MLE:
# 1. Introduce more complex concentration parameters
# 2. Add highly nonlinear relationships
# 3. Include multi-way interactions
# 4. Create hierarchical dependencies
# 5. Add step functions with smooth transitions
# Category 1: Complex hierarchical structure with smooth transitions
alpha[,1] <- pmax(1, 
                  exp(X[,1]))
# Category 2: Multi-way interactions with step functions
alpha[,2] <- pmax(1, 
                  3 * 
                    exp(X[,3]))
# Category 3: Highly nonlinear relationships with smooth transitions
alpha[,3] <- pmax(1, 
                  2 + 
                    X[,3] * exp(-X[,2]))

# Generate Dirichlet-distributed response Y
Y <- t(apply(alpha, 1, function(a) {
  return(as.numeric(rdirichlet(1, a)))  # Sample from Dirichlet distribution
}))
# Helper function for smooth transition
sigmoid <- function(x) {
  1 / (1 + exp(-x))
}
# Generate Dirichlet-distributed response Y
Y <- t(apply(alpha, 1, function(a) {
  return(as.numeric(rdirichlet(1, a)))
}))
n_train <- floor(0.80 * n_samples)
indices <- sample(1:n_samples, n_samples)
train_indices <- indices[1:n_train]
test_indices <- indices[(n_train + 1):n_samples]
# Create train datasets
X_train <- X[train_indices, ]
Y_train <- Y[train_indices, ]
# Create test datasets
X_test <- X[test_indices, ]
Y_test <- Y[test_indices, ]

# Install and load required packages
if (!require("JuliaCall")) install.packages("JuliaCall")
library(JuliaCall)


# Initialize Julia
julia_setup()

# Load required Julia packages
julia_command("using Random, Distributions, CSV, DataFrames, StatsBase, Statistics")

# Source your Julia prediction script
julia_source("C:\\Users\\29827094\\Documents\\GitHub\\DirichletRandomForest\\Julia\\rf_for_call_to_R.jl")


# Function to summarize results remains the same
summarize_results <- function(results) {
  # Create summary dataframe
  metrics <- c("aitchison_distance", "compositional_r2", "mean_rmse")
  summary_df <- data.frame(
    Model = character(),
    Metric = character(),
    Train = numeric(),
    Test = numeric(),
    stringsAsFactors = FALSE
  )
  
  models <- names(results)
  for (model in models) {
    if (!is.null(results[[model]])) {  # Only include successful models
      for (metric in metrics) {
        summary_df <- rbind(summary_df, data.frame(
          Model = model,
          Metric = metric,
          Train = if(metric == "mean_rmse") results[[model]]$train$mean_rmse else results[[model]]$train[[metric]],
          Test = if(metric == "mean_rmse") results[[model]]$test$mean_rmse else results[[model]]$test[[metric]]
        ))
      }
    }
  }
  
  # Print formatted summary
  cat("\nModel Comparison Results:\n")
  cat("=======================\n\n")
  
  for (model in unique(summary_df$Model)) {
    cat(sprintf("\n%s:\n", gsub("_", " ", toupper(model))))
    model_results <- summary_df[summary_df$Model == model,]
    for (i in 1:nrow(model_results)) {
      cat(sprintf("%s:\n", gsub("_", " ", model_results$Metric[i])))
      cat(sprintf("  Train: %.4f\n", model_results$Train[i]))
      cat(sprintf("  Test:  %.4f\n", model_results$Test[i]))
    }
    cat("\n")
  }
  
  # Create visualization
  if (nrow(summary_df) > 0) {
    library(ggplot2)
    
    plot_data <- reshape2::melt(summary_df, id.vars = c("Model", "Metric"))
    plot_data$Metric <- gsub("_", " ", plot_data$Metric)
    
    p <- ggplot(plot_data, aes(x = Model, y = value, fill = variable)) +
      geom_bar(stat = "identity", position = "dodge") +
      facet_wrap(~Metric, scales = "free_y") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(x = "Model", y = "Value", fill = "Dataset") +
      scale_fill_brewer(palette = "Set1")
    
    print(p)
  }
  
  return(summary_df)
}
###############
compare_models <- function(X_train, Y_train, X_test, Y_test) {
  results <- list()
 
  # 2. Compositional Linear Log-ratio Model
  tryCatch({
    # Transform data using centered log-ratio (CLR)
    Y_train_clr <- clr(as.matrix(Y_train))
    Y_test_clr <- clr(as.matrix(Y_test))
    
    # Fit Random Forest models for each CLR-transformed component
    rf_models <- lapply(1:ncol(Y_train_clr), function(i) {
      randomForest(
        x = X_train,
        y = Y_train_clr[,i],
        ntree = 500,
        mtry = floor(sqrt(ncol(X_train))),
        importance = TRUE
      )
    })
    
    # Make predictions and back-transform
    rf_train_pred <- do.call(cbind, lapply(rf_models, function(mod) predict(mod, X_train)))
    rf_test_pred <- do.call(cbind, lapply(rf_models, function(mod) predict(mod, X_test)))
    
    # Back-transform predictions to compositional space
    train_pred_comp <- clrInv(rf_train_pred)
    test_pred_comp <- clrInv(rf_test_pred)
    
    results$clr_rf <- list(
      train = evaluate_performance(as.data.frame(Y_train), as.data.frame(train_pred_comp)),
      test = evaluate_performance(as.data.frame(Y_test), as.data.frame(test_pred_comp)),
      models = rf_models
    )
  }, error = function(e) {
    warning("Error in CLR Random Forest Model: ", e$message)
    results$clr_rf <- NULL
  })
  
  # 3. Isometric Log-ratio Transform with Random Forest
  tryCatch({
    # Transform data using ILR
    Y_train_ilr <- ilr(as.matrix(Y_train))
    Y_test_ilr <- ilr(as.matrix(Y_test))
    
    # Fit random forest on ILR-transformed data
    ilr_models <- lapply(1:ncol(Y_train_ilr), function(i) {
      randomForest(Y_train_ilr[,i] ~ ., data = as.data.frame(X_train))
    })
    
    # Make predictions and back-transform
    ilr_train_pred <- do.call(cbind, lapply(ilr_models, function(mod) predict(mod, as.data.frame(X_train))))
    ilr_test_pred <- do.call(cbind, lapply(ilr_models, function(mod) predict(mod, as.data.frame(X_test))))
    
    # Back-transform predictions to compositional space
    train_pred_comp_ilr <- ilrInv(ilr_train_pred)
    test_pred_comp_ilr <- ilrInv(ilr_test_pred)
    
    results$ilr_rf <- list(
      train = evaluate_performance(as.data.frame(Y_train), as.data.frame(train_pred_comp_ilr)),
      test = evaluate_performance(as.data.frame(Y_test), as.data.frame(test_pred_comp_ilr))
    )
  }, error = function(e) {
    warning("Error in ILR Random Forest: ", e$message)
    results$ilr_rf <- NULL
  })
  # Fixed Dirichlet Regression section
  tryCatch({
    # Ensure proper column names for Y_train
    Y_train_dr <- as.data.frame(Y_train)
    colnames(Y_train_dr) <- paste0("Y", 1:ncol(Y_train))
    
    # Ensure proper column names for X_train
    X_train_df1 <- as.data.frame(X_train)
    if (is.null(colnames(X_train))) {
      colnames(X_train_df1) <- paste0("X", 1:ncol(X_train))
    }
    
    # Create DR_data object
    Y_comp_train <- DirichletReg::DR_data(Y_train_dr)
    
    # Combine into training dataset
    training_data <- cbind(X_train_df1, Y_comp = Y_comp_train)
    
    # Create formula string using only the actual X variables
    x_vars <- colnames(X_train_df1)
    formula_str <- paste("Y_comp ~", paste(x_vars, collapse = " + "))
    
    # Fit model
    dr_model <- DirichletReg::DirichReg(
      formula = as.formula(formula_str),
      data = training_data,
      model = "alternative",
      verbosity = 0
    )
    
    # Prepare test data
    X_test_df1 <- as.data.frame(X_test)
    colnames(X_test_df1) <- colnames(X_train_df1)
    
    # Make predictions
    train_pred_dr <- predict(dr_model, newdata = X_train_df1)
    test_pred_dr <- predict(dr_model, newdata = X_test_df1)
    
    # Store results
    results$dirichlet_reg <- list(
      train = evaluate_performance(Y_train, as.matrix(train_pred_dr)),
      test = evaluate_performance(Y_test, as.matrix(test_pred_dr)),
      model = dr_model,
      coefficients = coef(dr_model)
    )
    
  }, error = function(e) {
    warning("Error in Dirichlet Regression: ", e$message)
    print(e)
    results$dirichlet_reg <- NULL
  })
  
  return(results)
}

# Example usage with simulation data:
# First generate simulation data as you provided...
# Then run:
model_results <- compare_models(X_train, Y_train, X_test, Y_test)
summary <- summarize_results(model_results)
