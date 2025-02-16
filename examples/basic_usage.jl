# Example usage of DirichletRandomForest
using DirichletRandomForest

# Generate example data
X, Y = generate_compositional_data(1000, 10, 3)

# Split into train and test
n_train = 800
X_train, X_test = X[1:n_train, :], X[n_train+1:end, :]
Y_train, Y_test = Y[1:n_train, :], Y[n_train+1:end, :]

# Create and fit a forest with default parameters
forest = DirichletForest(100)  # 100 trees
fit_dirichlet_forest!(forest, X_train, Y_train, 10)  # q_threshold = 10

# Make predictions
Y_pred = predict_dirichlet_forest(forest, X_test)

# Evaluate performance
distance = aitchison_distance(Y_test, Y_pred)
r2 = compositional_r2(Y_test, Y_pred)
println("Aitchison distance: ", distance)
println("Compositional R²: ", r2)

# Print variable importance
println("\nVariable importance:")
for (i, imp) in enumerate(forest.importance)
    println("Feature $i: ", round(imp, digits=4))
end