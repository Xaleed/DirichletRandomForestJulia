# dirichlet_forest_parallel.jl
using Distributed

# Add worker processes if not already added
if nprocs() == 1
    addprocs(Sys.CPU_THREADS - 1)
end

@everywhere begin
    using Distributions
    using Random
    using Statistics
    using DataFrames
    using SpecialFunctions
    # Make sure this file is in the same directory or use full path
    include("MLE_vs_MoM.JL")

    # Utility Functions
    function dirichlet_loglik(y::Vector{Float64}, alpha::Vector{Float64})
        loglik = loggamma(sum(alpha)) - sum(loggamma.(alpha)) + sum((alpha .- 1) .* log.(y))
        return loglik
    end

    # Dirichlet Node Structure
    mutable struct DirichletNode
        split_var::Union{Nothing,Int}
        split_value::Union{Nothing,Float64}
        left_child::Union{Nothing,DirichletNode}
        right_child::Union{Nothing,DirichletNode}
        terminal::Bool
        predictions::Union{Nothing,Vector{Float64}}
        improvement::Float64
    end

    # Dirichlet Forest Structure
    mutable struct DirichletForest
        trees::Vector{DirichletNode}
        importance::Union{Nothing,Vector{Float64}}
        importancef::Union{Nothing,Vector{Float64}}
        n_categories::Union{Nothing,Int}
        importancef_df::Union{Nothing,Matrix{Float64}}
        importance_df::Union{Nothing,Matrix{Float64}}
        n_trees::Int
    end

    # Constructor functions
    DirichletNode() = DirichletNode(nothing, nothing, nothing, nothing, false, nothing, 0.0)
    DirichletForest(n_trees::Int=100) = DirichletForest(Vector{DirichletNode}(), nothing, nothing, nothing, nothing, nothing, n_trees)

    # Find best split for Dirichlet tree
    function find_best_split_dirichlet(X::Matrix{Float64}, Y::Matrix{Float64},
        q_threshold::Int, node_samples::Vector{Int}, mtry::Union{Nothing,Int}=nothing,
        optimization_method::Function=estimate_parameters_mom)

        mtry = isnothing(mtry) ? Int(round(size(X, 2) / 3)) : mtry

        n_samples = length(node_samples)
        best_decrease = -Inf
        best_var = nothing
        best_value = nothing

        alpha_parent = optimization_method(Y[node_samples, :])
        parent_loglik = sum([dirichlet_loglik(Y[i, :], alpha_parent) for i in node_samples])

        if n_samples >= 2
            possible_split_vars = sample(1:size(X, 2), min(mtry, size(X, 2)), replace=false)

            for var_id in possible_split_vars
                x_var = X[node_samples, var_id]
                unique_vals = sort(unique(x_var))

                split_points = if length(unique_vals) > q_threshold
                    probs = range(0, 1, length=q_threshold + 2)[2:q_threshold+1]
                    [quantile(x_var, p) for p in probs]
                else
                    unique_vals[2:end]
                end

                for split_val in split_points
                    left_idx = node_samples[x_var.<=split_val]
                    right_idx = node_samples[x_var.>split_val]

                    if length(left_idx) < 2 || length(right_idx) < 2
                        continue
                    end

                    alpha_left = optimization_method(Y[left_idx, :])
                    alpha_right = optimization_method(Y[right_idx, :])

                    left_loglik = sum([dirichlet_loglik(Y[i, :], alpha_left) for i in left_idx])
                    right_loglik = sum([dirichlet_loglik(Y[i, :], alpha_right) for i in right_idx])
                    decrease = (left_loglik + right_loglik) - parent_loglik

                    if decrease > best_decrease
                        best_decrease = decrease
                        best_var = var_id
                        best_value = split_val
                    end
                end
            end
        end
        
        return best_decrease > 0 ? (var_id=best_var, value=best_value, decrease=best_decrease) : nothing
    end

    # Grow a single Dirichlet tree
    function grow_dirichlet_tree(X::Matrix{Float64}, Y::Matrix{Float64},
        q_threshold::Int,
        max_depth::Int=10,
        min_node_size::Int=5,
        mtry::Union{Nothing,Int}=nothing,
        optimization_method::Function=estimate_parameters_mom)
        
        importance = zeros(size(X, 2))
        importancef = zeros(size(X, 2))
    
        function grow_node(node_samples::Vector{Int}, depth::Int=0)
            node = DirichletNode()
            
            if depth >= max_depth || length(node_samples) < min_node_size * 2
                node.terminal = true
                node.predictions = vec(mean(Y[node_samples, :], dims=1))
                return node
            end
    
            split = find_best_split_dirichlet(X, Y, q_threshold, node_samples, mtry, optimization_method)
    
            if isnothing(split)
                node.terminal = true
                node.predictions = vec(mean(Y[node_samples, :], dims=1))
                return node
            end
    
            node.split_var = split.var_id
            node.split_value = split.value
            node.improvement = split.decrease
            importance[split.var_id] += split.decrease
            importancef[split.var_id] += 1
    
            left_samples = node_samples[X[node_samples, split.var_id].<=split.value]
            right_samples = node_samples[X[node_samples, split.var_id].>split.value]
    
            node.left_child = grow_node(left_samples, depth + 1)
            node.right_child = grow_node(right_samples, depth + 1)
    
            return node
        end
    
        root = grow_node(collect(1:size(X, 1)))
        return (tree=root, importance=importance, importancef=importancef)
    end

    # Prediction functions
    function predict_dirichlet_tree(tree::DirichletNode, X::Matrix{Float64})
        function predict_sample(node::DirichletNode, x::Vector{Float64})
            if node.terminal
                return node.predictions
            end

            if x[node.split_var] <= node.split_value
                return predict_sample(node.left_child, x)
            else
                return predict_sample(node.right_child, x)
            end
        end

        n_samples = size(X, 1)
        n_categories = length(tree.terminal ? tree.predictions : predict_sample(tree, X[1, :]))
        predictions = zeros(n_samples, n_categories)

        for i in 1:n_samples
            predictions[i, :] .= predict_sample(tree, X[i, :])
        end

        predictions ./= sum(predictions, dims=2)
        return predictions
    end
end

# Parallel Forest Fitting Function
function fit_dirichlet_forest_parallel!(forest::DirichletForest, X::Matrix{Float64}, Y::Matrix{Float64},
    q_threshold::Int, max_depth::Int=10, min_node_size::Int=5,
    mtry::Union{Nothing,Int}=nothing,
    optimization_method::Function=estimate_parameters_mom)
    
    forest.importance = zeros(size(X, 2))
    forest.importancef = zeros(size(X, 2))
    forest.n_categories = size(Y, 2)

    # Parallelize tree growing
    tree_results = @distributed (vcat) for i in 1:forest.n_trees
        sample_size = Int(round(1 * size(X, 1)))
        boot_idx = sample(1:size(X, 1), sample_size, replace=true)
        
        grow_dirichlet_tree(
            X[boot_idx, :],
            Y[boot_idx, :], 
            q_threshold,
            max_depth,
            min_node_size,
            mtry,
            optimization_method
        )
    end

    # Aggregate results
    forest.trees = [result.tree for result in tree_results]
    importance_list = [result.importance for result in tree_results]
    importancef_list = [result.importancef for result in tree_results]

    forest.importancef_df = reduce(vcat, importancef_list')'
    forest.importance_df = reduce(vcat, importance_list')'
    
    forest.importance = vec(mean(forest.importance_df, dims=1))
    forest.importancef = vec(mean(forest.importancef_df, dims=1))

    return forest
end

function predict_dirichlet_forest(forest::DirichletForest, X::Matrix{Float64})
    n_samples = size(X, 1)
    tree_preds = [predict_dirichlet_tree(tree, X) for tree in forest.trees]

    avg_preds = zeros(n_samples, forest.n_categories)

    for i in 1:n_samples
        sample_preds = zeros(length(forest.trees), forest.n_categories)
        for (t, pred) in enumerate(tree_preds)
            sample_preds[t, :] .= pred[i, :]
        end
        avg_preds[i, :] .= vec(mean(sample_preds, dims=1))
    end

    avg_preds ./= sum(avg_preds, dims=2)
    return avg_preds
end

# To be used from R: a function to process matrix data
function process_matrix_data(x_train, y_train, x_test)
    # Convert to expected type
    x_train = convert(Matrix{Float64}, x_train)
    y_train = convert(Matrix{Float64}, y_train)
    x_test = convert(Matrix{Float64}, x_test)
    
    return x_train, y_train, x_test
end

# Cleanup function
function cleanup_workers()
    rmprocs(workers())
    return "Workers cleaned up"
end