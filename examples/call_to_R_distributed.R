# Clean previous connections and set thread count

# Initialize and verify
library(JuliaCall)

julia_setup()

# Load required Julia packages

set.seed(123)
X_train <- matrix(rnorm(100*10), nrow=100, ncol=10)
Y_train <- matrix(runif(100*3), nrow=100, ncol=3)
# Normalize Y_train rows to sum to 1 (compositional data)
Y_train <- Y_train / rowSums(Y_train)
X_test <- matrix(rnorm(20*10), nrow=20, ncol=10)
Y_test <- matrix(runif(20*3), nrow=20, ncol=3)
Y_test <- Y_test / rowSums(Y_test)

# Source your Julia prediction script - UPDATE TO PARALLEL VERSION
julia_assign("x_train", X_train)
julia_assign("y_train", Y_train)
julia_assign("x_test", X_test)

# UPDATE: Change to parallel version file path
julia_main_file <- "C:\\Users\\Khaled\\Documents\\GitHub\\DirichletRandomForestJulia\\src\\dirichlet_forest_ml_distributed.jl"
julia_main_file <-"C:\\Users\\29827094\\Documents\\GitHub\\DirichletRandomForestJulia\\src\\dirichlet_forest_ml_distributed.jl"
julia_source(julia_main_file)

# Print worker information
julia_eval('
println("Number of workers: ", nworkers())
#println("Worker IDs: ", workers())
')

time_taken <- system.time({
  julia_pred <- julia_eval('begin 
  x_train, y_train, x_test = process_matrix_data(x_train, y_train, x_test)
  
  # Initialize forest
  forest = DirichletForest(500)
  
  # Train forest using MLE-Newton method WITH PARALLEL PROCESSING
  println("Training forest with parallel processing...")
  fit_dirichlet_forest!(
    forest, 
    x_train, 
    y_train, 
    500000000,  # q_threshold 
    200000,     # max_depth
    5,          # min_node_size
    5,          # mtry
    estimate_parameters_mom #estimate_parameters_mle_newton  # optimization_method
  )
  
  println("Forest training completed!")
  forest  # Return the trained forest
  end')
})

# Print the output (julia_pred)
print("Training completed:")
print(julia_pred)

# Print the time taken for execution
cat("Time taken for parallel training:\n")
print(time_taken)

# Predict on TEST data
cat("Making predictions on test data...\n")
pred_test <- julia_eval('
    predict_dirichlet_forest(forest, x_test)
')

# Predict on TRAIN data  
cat("Making predictions on train data...\n")
pred_train <- julia_eval('
    predict_dirichlet_forest(forest, x_train)
')

# Calculate evaluation metrics
cat("Calculating evaluation metrics...\n")
metrics <- julia_eval('
begin
    # Test metrics
    test_aitchison = aitchison_distance(y_test, pred_test)
    test_r2 = compositional_r2(y_test, pred_test)
    
    # Train metrics  
    train_aitchison = aitchison_distance(y_train, pred_train)
    train_r2 = compositional_r2(y_train, pred_train)
    
    println("=== EVALUATION METRICS ===")
    println("Test Aitchison Distance: ", test_aitchison)
    println("Test Compositional R²: ", test_r2)
    println("Train Aitchison Distance: ", train_aitchison)
    println("Train Compositional R²: ", train_r2)
    
    # Return metrics as named tuple
    (test_aitchison=test_aitchison, test_r2=test_r2, 
     train_aitchison=train_aitchison, train_r2=train_r2)
end
')

# Print variable importance
cat("Variable importance:\n")
importance <- julia_eval('
begin
    println("=== VARIABLE IMPORTANCE ===")
    println("Importance scores: ", forest.importance)
    println("Importance frequency: ", forest.importancef)
    forest.importance
end
')

# Clean up Julia workers when done
cat("Cleaning up workers...\n")
julia_eval('cleanup_workers()')

cat("Analysis completed!\n")

# Display results in R
cat("\n=== FINAL RESULTS ===\n")
cat("Test predictions shape:", dim(pred_test), "\n")
cat("Train predictions shape:", dim(pred_train), "\n")
cat("Test Aitchison Distance:", metrics[1], "\n")
cat("Test R²:", metrics[2], "\n")
cat("Train Aitchison Distance:", metrics[3], "\n")
cat("Train R²:", metrics[4], "\n")