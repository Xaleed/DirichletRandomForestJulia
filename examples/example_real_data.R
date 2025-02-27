# Load required libraries and source file
source("C:\\Users\\29827094\\Documents\GitHub\\DirichletRandomForestJulia\\src\\comparision_models.R")

# Load real data
train_df <- read.csv("C:/Users/29827094/Documents/GitHub/DirichletRandomForest/data_set/train_data.csv")
test_df <- read.csv("C:/Users/29827094/Documents/GitHub/DirichletRandomForest/data_set/test_data.csv")

# Extract features and responses
X_train <- as.matrix(train_df[, 4:40])  # Features from column 4 to 40
Y_train <- as.matrix(train_df[, 1:3])   # First 3 columns are responses
X_test <- as.matrix(test_df[, 4:40])    # Features from column 4 to 40
Y_test <- as.matrix(test_df[, 1:3])     # First 3 columns are responses

# Create formula string based on actual feature names
feature_names <- colnames(X_train)
x_vars <- paste(feature_names, collapse = " + ")
feature_names <- paste0("X", 1:length(feature_names))
x_vars <- paste(feature_names, collapse = " + ")

formula_str <- paste("Y_comp ~", x_vars)


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

dr_results <- fit_dirichlet_regression(X_train, Y_train, X_test, Y_test)







feature_names <- colnames(X_train)
x_vars <- paste(feature_names, collapse = " + ")
feature_names <- paste0("X", 1:length(feature_names))
x_vars <- paste(feature_names, collapse = " + ")

formula_str <- paste("Y_comp ~", x_vars)


# Run model comparison
model_results <- compare_models(X_train, Y_train, X_test, Y_test)

# Summarize results
summary <- summarize_results(model_results)
print(summary)

# Print variable importance for Dirichlet Random Forest
print("Variable Importance:")
print(model_results$Dirichlet_RF$importance)
df = model_results$Dirichlet_RF$importance
df[order(df$frequency, decreasing = TRUE), ]
# Save results if needed
if (!is.null(model_results$drf$importance)) {
  write.csv(model_results$drf$importance, 
            "C:\\Users\\29827094\\Documents\\GitHub\\DirichletRandomForest\\R\\variable_importance.csv")
}

# Save summary results
if (!is.null(summary)) {
  write.csv(summary, 
            "C:\\Users\\29827094\\Documents\\GitHub\\DirichletRandomForest\\R\\model_summary.csv")
}