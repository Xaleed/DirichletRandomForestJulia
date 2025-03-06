

# Source your parallel Julia implementation

# Clean previous connections and set thread count
library(JuliaCall)
julia_setup()

# Set seed for reproducibility
set.seed(123)

# Generate sample data
X_train <- matrix(rnorm(100*10), nrow=100, ncol=10)
Y_train <- matrix(runif(100*3), nrow=100, ncol=3)
# Normalize Y_train rows to sum to 1 (compositional data)
Y_train <- Y_train / rowSums(Y_train)
X_test <- matrix(rnorm(20*10), nrow=20, ncol=10)
Y_test <- matrix(runif(20*3), nrow=20, ncol=3)
Y_test <- Y_test / rowSums(Y_test)

# Source the parallel implementation
julia_source("C:\\Users\\29827094\\Documents\\GitHub\\DirichletRandomForestJulia\\src\\rf_for_call_to_R.jl")
julia_source("C:\\Users\\29827094\\Documents\\GitHub\\DirichletRandomForestJulia\\src\\distribute_forest.jl")
# Transfer data to Julia
julia_assign("x_train", X_train)
julia_assign("y_train", Y_train)
julia_assign("x_test", X_test)

# Source the MLE_vs_MoM.JL file (ensure this file is in the correct path)
julia_command('include("C:/Users/29827094/Documents/GitHub/DirichletRandomForestJulia/src/MLE_vs_MoM.JL")')

# Run parallel training
time_taken <- system.time({
  julia_pred <- julia_eval('begin 
    # Process input data
    x_train, y_train, x_test = process_matrix_data(x_train, y_train, x_test)
    
    # Store processed data for later use
    global x_train_processed = x_train
    global x_test_processed = x_test
    
    # Initialize forest
    forest = DirichletForest(500)
    
    # Train forest using MLE-Newton method with parallel implementation
    println("Training forest in parallel...")
    fit_dirichlet_forest_parallel!(
      forest, 
      x_train, 
      y_train, 
      500000000,  
      200000,  
      5,   # min_node_size
      5,   # mtry
      estimate_parameters_mle_newton
    )
    
    "Forest training completed"
  end')
})

# Print the output (julia_pred)
print(julia_pred)
# Print the time taken for execution
print(time_taken)

# Predict on TEST data
pred_test <- julia_eval('
  predict_dirichlet_forest(forest, x_test_processed)
')

# Predict on TRAIN data
pred_train <- julia_eval('
  predict_dirichlet_forest(forest, x_train_processed)
')

# Clean up workers when done
julia_eval('cleanup_workers()')

