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
library(MASS)
####################
#keras::install_keras()

# SIMPLE DATASET
# Features: X1, X5 numerical; others categorical from {1,2,3,4,5}
# Simple linear relationships with minimal interactions



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
    sum(((y_true))^2)
  }))
  
  residual_var <- sum(sapply(1:n_samples, function(i) {
    y_true <- as.numeric(Y_true[i,])
    y_pred <- as.numeric(Y_pred[i,])
    
    y_true <- pmax(y_true, .Machine$double.eps)
    y_pred <- pmax(y_pred, .Machine$double.eps)
    
    gm_true <- geometric_mean(y_true)
    gm_pred <- geometric_mean(y_pred)
    
    sum(((y_true) - (y_pred))^2)
  }))
  
  comp_r2 <- 1 - residual_var/total_var
  rmse <- sqrt(colMeans((Y_true - Y_pred)^2))
  
  return(list(
    aitchison_distance = aitchison_dist,
    MEC = comp_r2,
    RMSE = rmse,
    mean_rmse = mean(rmse)
  ))
}


library(MASS)  # For rdirichlet function

# Install and load required packages
if (!require("JuliaCall")) install.packages("JuliaCall")
library(JuliaCall)


# Initialize Julia
julia_setup()

# Load required Julia packages
julia_command("using Random, Distributions, CSV, DataFrames, StatsBase, Statistics")

# Source your Julia prediction script
julia_source("C:\\Users\\29827094\\Documents\\GitHub\\DirichletRandomForestJulia\\src\\rf_for_call_to_R.jl")

#x_vars <- paste(paste0("X", 1:ncol(X_train)), collapse = " + ")
#formula_str <- paste("Y_comp ~", x_vars)
fit_dirichlet_regression <- function(X_train, Y_train, X_test, Y_test, model_type = "alternative") {
  # Create combined training dataframe
  train_df <- as.data.frame(cbind(Y_train, X_train))
  
  # Name columns
  colnames(train_df) <- c(paste0("Y", 1:ncol(Y_train)), 
                          paste0("X", 1:ncol(X_train)))
  
  # Create DR_data from just the Y columns
  y_cols <- paste0("Y", 1:ncol(Y_train))
  train_df$Y_comp <- DR_data(train_df[, y_cols])
  
  # Create formula

  
  # Fit model
  dr_model <- DirichReg(
    formula = as.formula(formula_str),
    data = train_df,
    model = model_type,
    verbosity = 0
  )
  
  # Prepare test data
  test_df <- as.data.frame(cbind(Y_test, X_test))
  colnames(test_df) <- c(paste0("Y", 1:ncol(Y_test)), 
                         paste0("X", 1:ncol(X_test)))
  test_df$Y_comp <- DR_data(test_df[, y_cols])
  
  # Make predictions
  train_pred <- predict(dr_model, newdata = train_df)
  test_pred <- predict(dr_model, newdata = test_df)
  
  return(list(
    model = dr_model,
    predictions = list(
      train = train_pred,
      test = test_pred
    )
  ))
}
# Fit the model




compare_models <- function(X_train, Y_train, X_test, Y_test) {
  results <- list()
  #Dirichlet Random Forest using Julia
  tryCatch({
    # Transfer data directly to Julia
    julia_assign("x_train", X_train)
    julia_assign("y_train", Y_train)
    julia_assign("x_test", X_test)
    
    # Process data and make predictions in Julia - fixed syntax
    julia_pred <- julia_eval('begin
      # Process the data
      x_train, y_train, x_test = process_matrix_data(x_train, y_train, x_test)
      
      # Make predictions
      pred_train, pred_test, importance, importance_freq = train_and_predict(
          x_train, 
          y_train, 
          x_test,
          n_trees=500,
          q_threshold=500000000,
          max_depth=200000,
          min_node_size=5
      )
      
      # Return results
      Dict(
          "pred_train" => pred_train,
          "pred_test" => pred_test,
          "importance" => importance,
          "importance_freq" => importance_freq
      )
    end')
    
    # Convert Julia predictions to R matrices
    predictions_train <- as.matrix(julia_pred$pred_train)
    predictions_test <- as.matrix(julia_pred$pred_test)
    
    # Create feature importance dataframe
    feature_importance <- data.frame(
      feature = colnames(X_train),
      importance = julia_pred$importance,
      frequency = julia_pred$importance_freq
    )
    
    # Evaluate performance
    results$Dirichlet_RF <- list(
      test = evaluate_performance(Y_test, predictions_test),
      train = evaluate_performance(Y_train, predictions_train),
      importance = feature_importance
    )
    
  }, error = function(e) {
    warning("Error in Dirichlet Random Forest: ", e$message)
    print(e)
    results$Dirichlet_RF <- NULL
  })
  # 1. Dirichlet Regression
  tryCatch({
    dr_results <- fit_dirichlet_regression(X_train, Y_train, X_test, Y_test)
    
    # Evaluate performance
    results$Dirichlet_regression <- list(
      train = evaluate_performance(as.data.frame(Y_train), 
                                   as.data.frame(dr_results$predictions$train)),
      test = evaluate_performance(as.data.frame(Y_test), 
                                  as.data.frame(dr_results$predictions$test)),
      model = dr_results$model
    )
  }, error = function(e) {
    warning("Error in Dirichlet Regression: ", e$message)
    results$Dirichlet_regression <- NULL
  })
  
  # 2. Compositional Linear Log-ratio Model (CLR + Random Forest)
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
    
    results$CLR_RF <- list(
      train = evaluate_performance(as.data.frame(Y_train), as.data.frame(train_pred_comp)),
      test = evaluate_performance(as.data.frame(Y_test), as.data.frame(test_pred_comp)),
      models = rf_models
    )
  }, error = function(e) {
    warning("Error in CLR Random Forest Model: ", e$message)
    results$CLR_RF <- NULL
  })
  
  # 3. Isometric Log-ratio Transform with Random Forest (ILR + Random Forest)
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
    
    results$ILR_RF <- list(
      train = evaluate_performance(as.data.frame(Y_train), as.data.frame(train_pred_comp_ilr)),
      test = evaluate_performance(as.data.frame(Y_test), as.data.frame(test_pred_comp_ilr))
    )
  }, error = function(e) {
    warning("Error in ILR Random Forest: ", e$message)
    results$ILR_RF <- NULL
  })
  
  return(results)
}




summarize_results <- function(results) {
  # Create summary dataframe
  metrics <- c("aitchison_distance", "MEC", "RMSE")
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
  
  # Create visualizations
  if (nrow(summary_df) > 0) {
    library(ggplot2)
    library(gridExtra)
    
    # Reshape data for plotting
    plot_data <- reshape2::melt(summary_df, id.vars = c("Model", "Metric"))
    plot_data$Metric <- gsub("_", " ", plot_data$Metric)
    
    # Plot 1: Bar plot for metrics
    p1 <- ggplot(plot_data, aes(x = Model, y = value, fill = variable)) +
      geom_bar(stat = "identity", position = "dodge") +
      facet_wrap(~Metric, scales = "free_y") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(x = "Model", y = "Value", fill = "Dataset") +
      scale_fill_brewer(palette = "Set1") +
      ggtitle("Model Performance Comparison")
    
    # Plot 2: Line plot for train vs. test performance
    p2 <- ggplot(plot_data, aes(x = Metric, y = value, color = Model, group = Model)) +
      geom_line(size = 1) +
      geom_point(size = 3) +
      facet_wrap(~variable, scales = "free_y") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(x = "Metric", y = "Value", color = "Model") +
      ggtitle("Train vs. Test Performance")
    
    # Plot 3: Heatmap of performance metrics
    p3 <- ggplot(plot_data, aes(x = Model, y = Metric, fill = value)) +
      geom_tile() +
      facet_wrap(~variable, scales = "free") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      scale_fill_gradient(low = "white", high = "steelblue") +
      labs(x = "Model", y = "Metric", fill = "Value") +
      ggtitle("Performance Heatmap")
    
    # Combine plots
    grid.arrange(p1, ncol = 1)
  }
  
  return(summary_df)
}

#############################################################################
#sellected


