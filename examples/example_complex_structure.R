source("C:\\Users\\29827094\\Documents\\GitHub\\DirichletRandomForestJulia\\src\\comparision_models.R")

library(ggplot2)
library(reshape2)



# Function to simulate Dirichlet-distributed data
simulate_dirichlet_data <- function(n_samples, n_features = 10, n_categories = 3, seed) {
  set.seed(seed)
  
  # Initialize feature matrix
  X <- matrix(0, nrow = n_samples, ncol = n_features)
  colnames(X) <- paste0("X", 1:n_features)
  
  # Generate categorical and continuous features
  for (col in 1:n_features) {
    X[, col] <- sample(1:5, n_samples, replace = TRUE)  # Categorical features
  }
  X[, 1] <- runif(n_samples)  # Continuous feature
  X[, 2] <- runif(n_samples)
  X[, 3] <- runif(n_samples)
  
  # Initialize alpha matrix
  alpha <- matrix(0, nrow = n_samples, ncol = n_categories)
  
  # Function to assign values based on quantiles
  assign_based_on_quantile <- function(x, quantiles = c(0.3, 0.6, 0.8), values = c(2, 3, 4)) {
    q <- quantile(x, probs = quantiles)
    result <- rep(1, length(x))
    result[x < q[1]] <- values[1]
    result[x >= q[1] & x < q[2]] <- values[2]
    result[x >= q[2] & x <= q[3]] <- values[3]
    return(result)
  }
  
  # Assign alpha values based on transformations of X
  alpha[, 1] <- assign_based_on_quantile(2 + 2 * sin(X[, 1]) + X[, 2]^3)
  alpha[, 2] <- assign_based_on_quantile(1 + 2.5 * exp(X[, 3]) + X[, 1]^2)
  alpha[, 3] <- assign_based_on_quantile(1 + (1 + X[, 2])^2 + X[, 3]^4)
  
  # Function to generate Dirichlet-like samples
  dirichlet_like_sample <- function(alpha_row) {
    samples <- sapply(alpha_row, function(a) max(0.001, rnorm(1, mean = max(0.1, a), sd = 0.5)))
    return(samples / sum(samples))
  }
  
  # Generate Y values
  Y <- t(apply(alpha, 1, dirichlet_like_sample))
  
  return(list(X = X, Y = Y, alpha = alpha))
}


# Function to split data into training and test sets
split_data <- function(X, Y, train_ratio = 0.8) {
  n_samples <- nrow(X)
  n_train <- floor(train_ratio * n_samples)
  indices <- sample(1:n_samples, n_samples)
  
  train_indices <- indices[1:n_train]
  test_indices <- indices[(n_train + 1):n_samples]
  
  return(list(
    X_train = X[train_indices, ],
    Y_train = Y[train_indices, ],
    X_test = X[test_indices, ],
    Y_test = Y[test_indices, ]
  ))
}
x_vars <- paste(paste0("X", 1:10), collapse = " + ")
formula_str <- paste("Y_comp ~", x_vars)
# Main iteration function
run_iteration <- function(seed, n_samples = 500) {
  set.seed(seed)
  
  tryCatch({
    # Simulate data
    sim_data <- simulate_dirichlet_data(n_samples, n_features = 10, n_categories = 3, seed)
    
    # Split data
    split_datasets <- split_data(sim_data$X, sim_data$Y)
    
    # Run models comparison
    
    results <- compare_models(
      split_datasets$X_train, 
      split_datasets$Y_train, 
      split_datasets$X_test, 
      split_datasets$Y_test
    )
    #x_vars <- paste(paste0("X", 1:ncol( split_datasets$X_train)), collapse = " + ")
    #formula_str <- paste("Y_comp ~", x_vars)
    # Process results
    metrics_list <- list()
    
    for (model_name in names(results)) {
      # Add training metrics
      if (!is.null(results[[model_name]]$train)) {
        metrics_list[[length(metrics_list) + 1]] <- data.frame(
          iteration = seed,
          model = model_name,
          dataset = "train",
          aitchison_distance = results[[model_name]]$train$aitchison_distance,
          MEC = results[[model_name]]$train$MEC,
          RMSE = results[[model_name]]$train$RMSE
        )
      }
      
      # Add test metrics
      if (!is.null(results[[model_name]]$test)) {
        metrics_list[[length(metrics_list) + 1]] <- data.frame(
          iteration = seed,
          model = model_name,
          dataset = "test",
          aitchison_distance = results[[model_name]]$test$aitchison_distance,
          MEC = results[[model_name]]$test$MEC,
          RMSE = results[[model_name]]$test$RMSE
        )
      }
    }
    
    return(do.call(rbind, metrics_list))
    
  }, error = function(e) {
    warning(paste("Error in iteration", seed, ":", e$message))
    return(data.frame(
      iteration = numeric(),
      model = character(),
      dataset = character(),
      aitchison_distance = numeric(),
      MEC = numeric(),
      RMSE = numeric()
    ))
  })
}




################
##########################
# Function to create and manage directories
setup_directories <- function(base_dir) {
  # Create main directory if it doesn't exist
  if (!dir.exists(base_dir)) {
    dir.create(base_dir, recursive = TRUE)
  }
  
  # Create subdirectories for different types of output
  subdirs <- c("plots", "data", "summaries")
  dirs <- sapply(file.path(base_dir, subdirs), function(d) {
    if (!dir.exists(d)) dir.create(d)
    return(d)
  })
  
  return(as.list(setNames(dirs, subdirs)))
}

# Function to generate filename with sample size
get_filename <- function(base_name, n_samples, extension) {
  sprintf("%s_%d%s", base_name, n_samples, extension)
}

# Function to generate and save plots
generate_plots <- function(all_results, dirs, n_samples) {
  # Calculate mean results
  mean_results <- aggregate(
    cbind(aitchison_distance, MEC, RMSE) ~ model + dataset,
    data = all_results,
    FUN = mean
  )
  
  # Create summary plot
  mean_results_long <- melt(mean_results, 
                            id.vars = c("model", "dataset"),
                            measure.vars = c("aitchison_distance", "MEC", "RMSE"))
  
  summary_plot <- ggplot(mean_results_long, aes(x = model, y = value, fill = dataset)) +
    geom_bar(stat = "identity", position = "dodge") +
    facet_wrap(~ variable, scales = "free_y") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = sprintf("Mean Performance Metrics by Model (n=%d)", n_samples),
         x = "Model", 
         y = "Value", 
         fill = "Dataset") +
    scale_fill_brewer(palette = "Set1")
  
  # Save summary plot with sample size in filename
  ggsave(
    file.path(dirs$plots, get_filename("summary_performance_mean", n_samples, ".pdf")), 
    summary_plot, width = 12, height = 8
  )
  
  # Create detailed performance plots
  results_long <- melt(all_results, 
                       id.vars = c("iteration", "model", "dataset"),
                       variable.name = "metric",
                       value.name = "value")
  
  # Training data plot
  train_plot <- ggplot(subset(results_long, dataset == "train"), 
                       aes(x = model, y = value, fill = model)) +
    geom_boxplot() +
    facet_wrap(~metric, scales = "free_y") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = sprintf("Training Set Performance (n=%d)", n_samples),
         x = "Model",
         y = "Value") +
    scale_fill_brewer(palette = "Set3")
  
  # Test data plot
  test_plot <- ggplot(subset(results_long, dataset == "test"), 
                      aes(x = model, y = value, fill = model)) +
    geom_boxplot() +
    facet_wrap(~metric, scales = "free_y") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = sprintf("Test Set Performance (n=%d)", n_samples),
         x = "Model",
         y = "Value") +
    scale_fill_brewer(palette = "Set3")
  
  # Calculate and plot differences
  results_wide <- dcast(results_long, iteration + model + metric ~ dataset, value.var = "value")
  results_wide$difference <- results_wide$test - results_wide$train
  
  diff_plot <- ggplot(results_wide, aes(x = model, y = difference, fill = model)) +
    geom_boxplot() +
    facet_wrap(~metric, scales = "free_y") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = sprintf("Performance Difference (Test - Train) (n=%d)", n_samples),
         x = "Model",
         y = "Difference") +
    scale_fill_brewer(palette = "Set3") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red")
  
  # Save all plots with sample size in filenames
  ggsave(
    file.path(dirs$plots, get_filename("train_performance", n_samples, ".pdf")), 
    train_plot, width = 12, height = 8
  )
  ggsave(
    file.path(dirs$plots, get_filename("test_performance", n_samples, ".pdf")), 
    test_plot, width = 12, height = 8
  )
  ggsave(
    file.path(dirs$plots, get_filename("performance_difference", n_samples, ".pdf")), 
    diff_plot, width = 12, height = 8
  )
  
  # Save summary statistics with sample size in filename
  summary_stats <- aggregate(
    cbind(aitchison_distance, MEC, RMSE) ~ model + dataset, 
    data = all_results,
    FUN = function(x) c(mean = mean(x), sd = sd(x), median = median(x))
  )
  
  write.csv(
    summary_stats, 
    file.path(dirs$summaries, get_filename("model_comparison_summary", n_samples, ".csv")), 
    row.names = FALSE
  )
}

# Main function to run simulation
run_simulation <- function(n_iterations = 40, 
                           n_samples = 500, 
                           seed_multiplier = 48,
                           base_dir = "simulation_results") {
  # Setup directories
  dirs <- setup_directories(base_dir)
  
  # Run iterations
  test_results <- lapply(1:n_iterations, function(i) {
    result <- run_iteration(seed_multiplier * i, n_samples)
    if (nrow(result) == 0) {
      warning(paste("Iteration", i, "returned no results"))
    }
    return(result)
  })
  
  # Combine results
  valid_results <- test_results[sapply(test_results, nrow) > 0]
  if (length(valid_results) == 0) {
    stop("No valid results were generated")
  }
  
  all_results <- do.call(rbind, valid_results)
  
  # Save raw results with sample size in filename
  write.csv(
    all_results, 
    file.path(dirs$data, get_filename("raw_results", n_samples, ".csv")), 
    row.names = FALSE
  )
  
  # Generate and save plots
  generate_plots(all_results, dirs, n_samples)
  
  # Return results and directory information
  return(list(
    results = all_results,
    directories = dirs
  ))
}

# Function to run simulations for multiple sample sizes
run_multiple_simulations <- function(sample_sizes = c(100, 500, 1000), 
                                     n_iterations = 40,
                                     base_dir = "simulation_results") {
  results_list <- list()
  
  for (n in sample_sizes) {
    cat(sprintf("\nRunning simulation for n=%d samples...\n", n))
    results_list[[as.character(n)]] <- run_simulation(
      n_iterations = n_iterations,
      n_samples = n,
      base_dir = base_dir
    )
  }
  
  return(results_list)
}

# Example usage:
#Run simulations for different sample sizes
sample_sizes <- c(100,200, 500, 2000)
all_results <- run_multiple_simulations(
  sample_sizes = sample_sizes,
  n_iterations = 40,
  base_dir = "C:\\Users\\29827094\\Documents\GitHub\\DirichletRandomForestJulia\\Results_complex_structure\\"
)

