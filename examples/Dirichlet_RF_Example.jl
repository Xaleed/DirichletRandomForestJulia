# Example application of Dirichlet Forest
using Distributions
using Random
using Statistics
using DataFrames

# Set random seed for reproducibility
Random.seed!(123)

# Include your main Dirichlet Forest code
# include("dirichlet_forest.jl")  # Uncomment this line to load your main code

# Function to generate synthetic dataset
function generate_dirichlet_data(n_samples::Int)
    # Generate three feature variables X1, X2, X3
    X = rand(n_samples, 3)
    
    # Initialize target matrix
    Y = zeros(n_samples, 3)
    
    # Generate Dirichlet distributed targets based on your specification
    for i in 1:n_samples
        # Define alpha parameters based on features
        α1 = 1 + 2 * X[i, 1]
        α2 = 1 + 0.5 * X[i, 2] 
        α3 = 1 + X[i, 3]
        
        # Generate Dirichlet sample
        alphas = [α1, α2, α3]
        Y[i, :] = rand(Dirichlet(alphas))
    end
    
    return X, Y
end

# Generate training and test datasets
println("Generating datasets...")
n_train = 1000
n_test = 200

X_train, Y_train = generate_dirichlet_data(n_train)
X_test, Y_test = generate_dirichlet_data(n_test)





include("C:\\Users\\Khaled\\Documents\\GitHub\\DirichletRandomForestJulia\\src\\dirichlet_forest_ml.jl")
forest = DirichletForest(100)  # 100 trees

# Fit the forest with your parameters
fit_dirichlet_forest!(
    forest, 
    X_train, 
    Y_train,
    5,        # Number of quantiles for splitting
    10,          # Maximum tree depth
    5,       # Minimum samples per node
    nothing,          # Use default (p/3 features per split)
    estimate_parameters_mom  # Method of moments
)



println("Training completed!")

# Make predictions on test set
println("\nMaking predictions on test set...")
Y_pred = predict_dirichlet_forest(forest, X_test)

# Display first few predictions vs actual
println("\nFirst 5 test predictions vs actual:")
println("Predicted:")
display(Y_pred[1:5, :])
println("Actual:")
display(Y_test[1:5, :])
println("Predicted row sums: ", sum(Y_pred[1:5, :], dims=2))

# Evaluate model performance
println("\nModel Performance:")
aitchison_dist = aitchison_distance(Y_test, Y_pred)
comp_r2 = compositional_r2(Y_test, Y_pred)

println("Aitchison Distance: ", round(aitchison_dist, digits=4))
println("Compositional R²: ", round(comp_r2, digits=4))

# Feature importance
println("\nFeature Importance:")
println("Mean Decrease in Log-likelihood:")
for i in 1:length(forest.importance)
    println("X$i: ", round(forest.importance[i], digits=4))
end

println("\nMean Decrease in Node Impurity (frequency):")
for i in 1:length(forest.importancef)
    println("X$i: ", round(forest.importancef[i], digits=4))
end

# Example of making predictions on new data
println("\n" * "="^50)
println("PREDICTION EXAMPLE ON NEW DATA")
println("="^50)

# Generate some new data points for prediction
println("\nGenerating 5 new data points...")
X_new = rand(5, 3)
println("New features:")
display(X_new)

# Make predictions
Y_new_pred = predict_dirichlet_forest(forest, X_new)
println("\nPredicted compositions:")
display(Y_new_pred)
println("Row sums: ", sum(Y_new_pred, dims=2))

# For comparison, show what the true Dirichlet parameters would be
println("\nTrue underlying Dirichlet parameters for these points:")
for i in 1:5
    α1 = 1 + 2 * X_new[i, 1]
    α2 = 1 + 0.5 * X_new[i, 2] 
    α3 = 1 + X_new[i, 3]
    
    # Expected values of Dirichlet distribution
    alpha_sum = α1 + α2 + α3
    expected_y1 = α1 / alpha_sum
    expected_y2 = α2 / alpha_sum
    expected_y3 = α3 / alpha_sum
    
    println("Point $i: α=[$α1, $α2, $α3] → Expected Y=[$expected_y1, $expected_y2, $expected_y3]")
end

# Visualize feature importance if you have plotting capabilities
println("\n" * "="^50)
println("FEATURE IMPORTANCE ANALYSIS")
println("="^50)

# Normalized importance scores
total_importance = sum(forest.importance)
normalized_importance = forest.importance ./ total_importance

println("Normalized Feature Importance (Log-likelihood decrease):")
for i in 1:length(normalized_importance)
    println("X$i: ", round(normalized_importance[i] * 100, digits=2), "%")
end

# Based on your data generation process:
# X1 should be most important (coefficient 2.0)
# X3 should be moderately important (coefficient 1.0) 
# X2 should be least important (coefficient 0.5)
println("\nExpected ranking based on data generation:")
println("X1 (coef=2.0) > X3 (coef=1.0) > X2 (coef=0.5)")

actual_ranking = sortperm(forest.importance, rev=true)
println("Actual ranking by model: X", join(actual_ranking, " > X"))

println("\nExample completed successfully!")