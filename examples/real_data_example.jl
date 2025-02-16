using CSV
using DataFrames
using Random
using Plots
using StatsPlots

# Include the necessary files from src
include("../src/dirichlet_forest.jl")  # Main implementation
include("../src/evaluation.jl")        # Evaluation metrics

function load_and_preprocess_data(file_path::String)
    # Read the data
    data = CSV.read(file_path, DataFrame)
    
    # Ensure the columns exist
    if !all(in.(["clay", "silt", "sand"], Ref(names(data))))
        error("Data must contain 'clay', 'silt', and 'sand' columns.")
    end
    
    # Normalize compositional data
    Y = Matrix{Float64}(data[:, ["clay", "silt", "sand"]])
    Y_normalized = Y ./ sum(Y, dims=2)  # Normalize rows to sum to 1
    
    # Extract features (all columns except the first 3)
    X = Matrix{Float64}(data[:, 4:end])
    
    return X, Y_normalized, names(data)[4:end]
end

function analyze_dirichlet_forest(X::Matrix{Float64}, Y::Matrix{Float64}, 
                                feature_names::Vector{String}, n_trees::Int=50)
    # Split data
    Random.seed!(42)
    n_samples = size(X, 1)
    train_idx = sample(1:n_samples, Int(0.7 * n_samples), replace=false)
    test_idx = setdiff(1:n_samples, train_idx)
    
    X_train = X[train_idx, :]
    Y_train = Y[train_idx, :]
    X_test = X[test_idx, :]
    Y_test = Y[test_idx, :]
    
    # Train forest
    forest = DirichletForest(n_trees)
    
    println("Training Dirichlet Random Forest...")
    @time fit_dirichlet_forest!(forest, X_train, Y_train, 20, 5, 10)
    
    # Make predictions
    predictions = predict_dirichlet_forest(forest, X_test)
    
    # Calculate metrics
    mse = mean((Y_test .- predictions).^2, dims=1)
    aitchison = aitchison_distance(Y_test, predictions)
    comp_r2 = compositional_r2(Y_test, predictions)
    
    # Create importance DataFrame
    importance_df = DataFrame(
        Feature = feature_names,
        Importance = forest.importance,
        FrequencyImportance = forest.importancef/forest.n_trees
    )
    sort!(importance_df, :Importance, rev=true)
    
    # Plot feature importance
    p1 = bar(importance_df.Feature, importance_df.Importance,
        xticks=(1:length(feature_names), feature_names),
        title="Log-likelihood Importance",
        xrotation=45,
        legend=false)
        
    importance_data = transpose(forest.importancef_df)
    p2 = boxplot(importance_data,
        xticks=(1:length(feature_names), feature_names),
        title="Frequency Importance",
        xrotation=45,
        legend=false)
    
    p = plot(p1, p2, layout=(2,1), size=(800,1000))
    
    return Dict(
        "forest" => forest,
        "importance_df" => importance_df,
        "metrics" => Dict(
            "mse" => vec(mse),
            "aitchison" => aitchison,
            "comp_r2" => comp_r2
        ),
        "plots" => p
    )
end

function run_real_data_analysis(file_path::String)
    # Load and preprocess data
    println("Loading data from: ", file_path)
    X, Y, feature_names = load_and_preprocess_data(file_path)
    
    # Run analysis
    results = analyze_dirichlet_forest(X, Y, feature_names)
    
    # Print results
    println("\nAnalysis Results:")
    println("----------------")
    println("MSE by category: ", join(round.(results["metrics"]["mse"], digits=4), ", "))
    println("Aitchison distance: ", round(results["metrics"]["aitchison"], digits=4))
    println("Compositional R²: ", round(results["metrics"]["comp_r2"], digits=4))
    
    println("\nTop 10 Important Features:")
    println(first(results["importance_df"], 10))
    
    # Display plots
    display(results["plots"])
    
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Replace with your actual data file path
    file_path = "data/soil_data.csv"
    results = run_real_data_analysis(file_path)
end