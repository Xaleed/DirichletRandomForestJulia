# dirichlet_forest.jl
using Distributions
using Random
using Statistics
using DataFrames
using SpecialFunctions
include("MLE_vs_MoM.JL")
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

# Utility functions
function dirichlet_loglik(y::Vector{Float64}, alpha::Vector{Float64})
    loglik = loggamma(sum(alpha)) - sum(loggamma.(alpha)) + sum((alpha .- 1) .* log.(y))
    return loglik
end


# Add optimization_method parameter to grow_dirichlet_tree
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
        function compute_mean_samples(samples::Vector{Int})
            # Calculate mean of Y values for samples in the node
            return mean(Y[samples, :], dims=1)
        end
        if depth >= max_depth || length(node_samples) < min_node_size * 2
            node.terminal = true
            #node.predictions = optimization_method(Y[node_samples, :])
            node.predictions =  vec(mean(Y[node_samples, :], dims=1))
            return node
        end

        # Modified find_best_split_dirichlet to use optimization_method
        split = find_best_split_dirichlet(X, Y, q_threshold, node_samples, mtry, optimization_method)

        if isnothing(split)
            node.terminal = true
            #node.predictions = optimization_method(Y[node_samples, :])
            node.predictions =  vec(mean(Y[node_samples, :], dims=1))
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

# Modified find_best_split_dirichlet to use optimization_method
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
    #isfinite(best_decrease)
    #best_decrease > 0
    return best_decrease > 0 ? (var_id=best_var, value=best_value, decrease=best_decrease) : nothing
end

# Modified fit_dirichlet_forest! to include optimization_method
function fit_dirichlet_forest!(forest::DirichletForest, X::Matrix{Float64}, Y::Matrix{Float64},
    q_threshold::Int, max_depth::Int=10, min_node_size::Int=5,
    mtry::Union{Nothing,Int}=nothing,
    optimization_method::Function=estimate_parameters_mom)
    forest.importance = zeros(size(X, 2))
    forest.importancef = zeros(size(X, 2))
    forest.n_categories = size(Y, 2)

    importance_list = Vector{Vector{Float64}}()
    importancef_list = Vector{Vector{Float64}}()
    sample_size = Int(round(1 * size(X, 1)))
    for i in 1:forest.n_trees
        boot_idx = sample(1:size(X, 1), sample_size, replace=false)
        tree_result = grow_dirichlet_tree(
            X[boot_idx, :],
            Y[boot_idx, :], q_threshold,
            max_depth,
            min_node_size,
            mtry,
            optimization_method
        )
        push!(forest.trees, tree_result.tree)
        push!(importance_list, tree_result.importance)
        push!(importancef_list, tree_result.importancef)
        forest.importance .+= tree_result.importance
        forest.importancef .+= tree_result.importancef
    end

    forest.importancef_df = reduce(vcat, importancef_list')'
    forest.importance_df = reduce(vcat, importance_list')'
    forest.importance ./= forest.n_trees
    forest.importancef ./= forest.n_trees

    return forest
end


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

# Evaluation metrics
function aitchison_distance(y_true::Matrix{Float64}, y_pred::Matrix{Float64})
    n_samples = size(y_true, 1)
    distances = zeros(n_samples)

    for i in 1:n_samples
        clr_true = log.(y_true[i, :]) .- mean(log.(y_true[i, :]))
        clr_pred = log.(y_pred[i, :]) .- mean(log.(y_pred[i, :]))
        distances[i] = sqrt(sum((clr_true .- clr_pred) .^ 2))
    end

    return mean(distances)
end

function compositional_r2(y_true::Matrix{Float64}, y_pred::Matrix{Float64})
    clr_true = log.(y_true) .- mean(log.(y_true), dims=2)
    clr_pred = log.(y_pred) .- mean(log.(y_pred), dims=2)

    total_var = sum((clr_true .- mean(clr_true, dims=1)) .^ 2)
    residual_var = sum((clr_true .- clr_pred) .^ 2)

    return 1 - residual_var / total_var
end


