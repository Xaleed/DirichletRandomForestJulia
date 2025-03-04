library(JuliaCall)


# Initialize Julia
julia_setup()

# Load required Julia packages
julia_command("using Random, Distributions, CSV, DataFrames, StatsBase, Statistics")
set.seed(123)
X_train <- matrix(rnorm(100*5), nrow=100, ncol=5)
Y_train <- matrix(runif(100*3), nrow=100, ncol=3)
# Normalize Y_train rows to sum to 1 (compositional data)
Y_train <- Y_train / rowSums(Y_train)

X_test <- matrix(rnorm(20*5), nrow=20, ncol=5)
Y_test <- matrix(runif(20*3), nrow=20, ncol=3)
Y_test <- Y_test / rowSums(Y_test)

# Source your Julia prediction script
julia_source("C:\\Users\\29827094\\Documents\\GitHub\\DirichletRandomForestJulia\\src\\rf_for_call_to_R.jl")
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

##################
#######################
##################################
# Save this as simulation_predictor.jl
julia_source("C:\\Users\\29827094\\Documents\\GitHub\\DirichletRandomForestJulia\\src\\dirichlet_forest_ml.jl")

julia_pred <- julia_eval('begin 
x_train, y_train, x_test = process_matrix_data(x_train, y_train, x_test)

# Initialize forest
forest = DirichletForest(500)

# Train forest using MLE-Newton method
println("Training forest...")
fit_dirichlet_forest!(
  forest, 
  x_train, 
  y_train, 
  500000000, 
  200000, 
  5,
  5,
  estimate_parameters_mle_newton
)
end')

# Predict on TEST data
pred_test <- julia_eval('
    predict_dirichlet_forest(forest, x_test_processed)
')

# Predict on TRAIN data
pred_train <- julia_eval('
    predict_dirichlet_forest(forest, x_train_processed)
')
