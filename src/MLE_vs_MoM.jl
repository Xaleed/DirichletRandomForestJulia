# Required imports
using Statistics  # for mean, var
using Distributions  # for Dirichlet distribution
using LinearAlgebra  # for norm
using SpecialFunctions  # for digamma, trigamma
using Optim  # for optimization methods
using NLopt  # for BOBYQA
using ForwardDiff  # for automatic differentiation

# Helper function for log-likelihood calculation
function dirichlet_loglik(y::Vector{Float64}, alpha::Vector{Float64})
    loglik = loggamma(sum(alpha)) - sum(loggamma.(alpha)) + sum((alpha .- 1) .* log.(y))
    return loglik
end

# Compare all methods
# function compare_estimation_methods( Y::Matrix{Float64})
#     methods = [
#         ("MLE-GBDT", estimate_parameters_mle_gbdt),
#         ("MoM", estimate_parameters_mom),
#         ("MLE-BFGS", estimate_parameters_mle_bfgs),
#         ("MLE-Nelder-Mead", estimate_parameters_mle_nelder_mead),
#         ("MLE-BOBYQA", estimate_parameters_mle_bobyqa),
#         ("MLE-Newton", estimate_parameters_mle_newton)
#     ]
    
#     results = Dict()
    
#     for (name, method) in methods
#         # Time the estimation
#         time = @elapsed alpha = method(Y)
        
#         # Calculate log-likelihood
#         loglik = sum(dirichlet_loglik(Y[i,:], alpha) for i in 1:size(Y,1))
        
#         # Calculate convergence rate
#         conv_rate = name == "MLE-Newton" ? 
#             "Quadratic" : "Linear/Superlinear"
        
#         # Store results
#         results[name] = (
#             alpha = alpha,
#             loglik = loglik,
#             time = time,
#             convergence = conv_rate
#         )
#     end
    
#     return results
# end

# Print comparison results
function print_comparison_results(results)
    println("\nMethod Comparison Results:")
    println("=" ^ 50)
    
    for (method, result) in results
        println("\n$method:")
        println("-" ^ 30)
        println("Parameters: ", round.(result.alpha, digits=4))
        println("Log-likelihood: ", round(result.loglik, digits=4))
        println("Computation time: ", round(result.time, digits=4), " seconds")
        println("Convergence rate: ", result.convergence)
    end
end
################
# Modified MoM estimation with bounds and scaling
function estimate_parameters_mom(Y::Matrix{Float64})
    means = vec(mean(Y, dims=1))
    variances = vec(var(Y, dims=1))
    
    # Ensure variance is not too close to zero
    min_var = 1e-6
    variances = max.(variances, min_var)
    
    # Modified MoM estimator with bounds
    v = means[1] * (1 - means[1]) / variances[1] - 1
    v = max(v, 0.1)  # Ensure positive concentration
    alpha = means * v
    
    # Ensure reasonable bounds
    alpha = clamp.(alpha, 0.1, 1000.0)
    
    return alpha
end

# Modified negative log-likelihood function with bounds checking
function neg_loglik(alpha::Vector{Float64}, Y::Matrix{Float64})
    if any(α -> α <= 0.1 || α > 1000.0, alpha)
        return Inf
    end
    
    n_samples = size(Y, 1)
    total = n_samples * (sum(-loggamma.(alpha)) + loggamma(sum(alpha)))
    
    for i in 1:n_samples
        if any(y -> y <= 0 || y >= 1, Y[i,:])
            return Inf
        end
        total += sum((alpha .- 1) .* log.(Y[i,:]))
    end
    
    return -total
end
# function estimate_parameters_mle_gbdt(
#     Y::Matrix{Float64};
#     learning_rate::Float64 = 0.1,
#     max_iterations::Int = 1000,
#     convergence_tolerance::Float64 = 1e-6
# # )
#     # Initial estimate using Method of Moments
#     function initial_estimate(Y)
#         means = vec(mean(Y, dims=1))
#         variances = vec(var(Y, dims=1))
        
#         # Compute concentration parameter
#         v = means[1] * (1 - means[1]) / variances[1] - 1
#         v = max(v, 0.1)
        
#         return means * v
#     end
    
#     # Log-likelihood calculation
#     function dirichlet_loglik(y::Vector{Float64}, alpha::Vector{Float64})
#         loglik = loggamma(sum(alpha)) - sum(loggamma.(alpha)) + sum((alpha .- 1) .* log.(y))
#         return loglik
#     end
    
#     # Gradient computation
#     function compute_gradient(alpha::Vector{Float64}, Y::Matrix{Float64})
#         n_samples = size(Y, 1)
#         digamma_sum = digamma(sum(alpha))
        
#         gradient = zeros(length(alpha))
#         for j in 1:length(alpha)
#             gradient[j] = n_samples * (digamma_sum - digamma(alpha[j]))
#             for i in 1:n_samples
#                 gradient[j] += log(Y[i,j])
#             end
#         end
        
#         return gradient
#     end
    
#     # Hessian computation
#     function compute_hessian(alpha::Vector{Float64})
#         n_samples = size(Y, 1)
#         n_categories = length(alpha)
        
#         trigamma_sum = trigamma(sum(alpha))
#         H = zeros(n_categories, n_categories)
        
#         for i in 1:n_categories
#             for j in 1:n_categories
#                 if i == j
#                     H[i,j] = n_samples * (-trigamma(alpha[i]) + trigamma_sum)
#                 else
#                     H[i,j] = n_samples * trigamma_sum
#                 end
#             end
#         end
        
#         return H
#     end
    
#     # Constrain optimization
#     function constrain_params(alpha::Vector{Float64})
#         return clamp.(alpha, 0.1, 100.0)
#     end
    
#     # Main optimization loop
#     alpha = initial_estimate(Y)
    
#     for iter in 1:max_iterations
#         gradient = compute_gradient(alpha, Y)
#         hessian = compute_hessian(alpha)
        
#         # Regularized Newton update
#         delta = -inv(hessian + 1e-6 * I) * gradient
        
#         # Line search with adaptive step
#         step = 1.0
#         while step > 1e-10
#             alpha_new = alpha + step * delta
#             alpha_new = constrain_params(alpha_new)
            
#             if all(α -> 0.1 < α < 100.0, alpha_new)
#                 alpha = alpha_new
#                 break
#             end
#             step *= 0.5
#         end
        
#         # Convergence check
#         if norm(gradient) < convergence_tolerance
#             break
#         end
#     end
    
#     return constrain_params(alpha)
# end

function estimate_parameters_mle_gbdt(
    Y::Matrix{Float64};
    learning_rate::Float64 = 0.01,
    n_estimators::Int = 100,
    subsample_ratio::Float64 = 0.8,
    convergence_tolerance::Float64 = 1e-6
)
    # Initial estimate using Method of Moments
    function initial_estimate(Y)
        means = vec(mean(Y, dims=1))
        variances = vec(var(Y, dims=1))
        v = means[1] * (1 - means[1]) / variances[1] - 1
        v = max(v, 0.1)
        return means * v
    end
   
    # Log-likelihood calculation
    function dirichlet_loglik(y::Vector{Float64}, alpha::Vector{Float64})
        loglik = loggamma(sum(alpha)) - sum(loggamma.(alpha)) + sum((alpha .- 1) .* log.(y))
        return loglik
    end
   
    # Pseudo-residuals computation (gradient)
    function compute_pseudo_residuals(Y::Matrix{Float64}, current_alpha::Vector{Float64})
        n_samples = size(Y, 1)
        digamma_sum = digamma(sum(current_alpha))
        
        residuals = zeros(n_samples, length(current_alpha))
        for j in 1:length(current_alpha)
            for i in 1:n_samples
                residuals[i,j] = digamma_sum - digamma(current_alpha[j]) + log(Y[i,j])
            end
        end
        return residuals
    end
    
    # Fit weak learner (using mean of residuals with subsampling)
    function fit_weak_learner(residuals::Matrix{Float64}, subsample_ratio::Float64)
        n_samples = size(residuals, 1)
        n_subsample = round(Int, n_samples * subsample_ratio)
        
        # Random subsampling
        subsample_indices = rand(1:n_samples, n_subsample)
        subsampled_residuals = residuals[subsample_indices, :]
        
        # Simple average as weak learner
        return vec(mean(subsampled_residuals, dims=1))
    end
    
    # Parameter constraints
    function constrain_params(alpha::Vector{Float64})
        return clamp.(alpha, 0.1, 100.0)
    end
    
    # Main boosting loop
    alpha = initial_estimate(Y)
    n_samples = size(Y, 1)
    
    for iter in 1:n_estimators
        # Compute pseudo-residuals
        residuals = compute_pseudo_residuals(Y, alpha)
        
        # Fit weak learner
        update = fit_weak_learner(residuals, subsample_ratio)
        
        # Update with learning rate
        alpha_new = alpha + learning_rate * update
        alpha_new = constrain_params(alpha_new)
        
        # Check convergence
        if norm(alpha_new - alpha) < convergence_tolerance
            alpha = alpha_new
            break
        end
        
        alpha = alpha_new
    end
    
    return constrain_params(alpha)
end
# Modified BFGS estimation
function estimate_parameters_mle_bfgs(Y::Matrix{Float64})
    n_categories = size(Y, 2)
    
    # Define objective function that only takes one argument
    function f(alpha)
        if any(α -> α <= 0.1 || α > 1000.0, alpha)
            return Inf
        end
        
        n_samples = size(Y, 1)
        total = n_samples * (sum(-loggamma.(alpha)) + loggamma(sum(alpha)))
        
        for i in 1:n_samples
            if any(y -> y <= 0 || y >= 1, Y[i,:])
                return Inf
            end
            total += sum((alpha .- 1) .* log.(Y[i,:]))
        end
        
        return -total
    end
    
    # Initial guess using MoM
    initial_alpha = estimate_parameters_mom(Y)
    
    # Optimize using BFGS with automatic differentiation
    result = Optim.optimize(
        f,
        initial_alpha,
        Optim.BFGS(),
        Optim.Options(
            x_tol = 1e-6,
            f_tol = 1e-6,
            iterations = 1000,
            show_trace = false
        )
    )
    
    return clamp.(Optim.minimizer(result), 0.1, 1000.0)
end

# Modified Nelder-Mead estimation
function estimate_parameters_mle_nelder_mead(Y::Matrix{Float64})
    initial_alpha = estimate_parameters_mom(Y)
    
    result = Optim.optimize(
        α -> neg_loglik(α, Y),
        initial_alpha,
        Optim.NelderMead(),
        Optim.Options(
            x_tol = 1e-6,
            f_tol = 1e-6,
            iterations = 1000,
            show_trace = false
        )
    )
    
    return clamp.(Optim.minimizer(result), 0.1, 1000.0)
end

# Modified BOBYQA estimation
function estimate_parameters_mle_bobyqa(Y::Matrix{Float64})
    n_categories = size(Y, 2)
    
    opt = NLopt.Opt(:LN_BOBYQA, n_categories)
    opt.min_objective = (x, g) -> neg_loglik(x, Y)
    opt.lower_bounds = fill(0.1, n_categories)
    opt.upper_bounds = fill(1000.0, n_categories)
    opt.xtol_rel = 1e-6
    opt.maxeval = 1000
    
    initial_alpha = estimate_parameters_mom(Y)
    
    (minf, minx, ret) = NLopt.optimize(opt, initial_alpha)
    
    return clamp.(minx, 0.1, 1000.0)
end

# Modified Newton-Raphson estimation
function estimate_parameters_mle_newton(Y::Matrix{Float64}; 
    max_iter::Int=1000, 
    tol::Float64=1e-6)
    
    n_samples = size(Y, 1)
    n_categories = size(Y, 2)
    
    function gradient(alpha)
        grad = zeros(n_categories)
        digamma_sum = digamma(sum(alpha))
        
        for j in 1:n_categories
            grad[j] = n_samples * (digamma_sum - digamma(alpha[j]))
            for i in 1:n_samples
                grad[j] += log(Y[i,j])
            end
        end
        
        return grad
    end
    
    function hessian(alpha)
        H = zeros(n_categories, n_categories)
        trigamma_sum = trigamma(sum(alpha))
        
        for i in 1:n_categories
            for j in 1:n_categories
                if i == j
                    H[i,j] = n_samples * (-trigamma(alpha[i]) + trigamma_sum)
                else
                    H[i,j] = n_samples * trigamma_sum
                end
            end
        end
        
        return H
    end
    
    # Initial guess using MoM
    alpha = estimate_parameters_mom(Y)
    
    for iter in 1:max_iter
        grad = gradient(alpha)
        H = hessian(alpha)
        
        # Compute update with regularization
        lambda = 1e-6
        H_reg = H + lambda * I
        delta = -H_reg \ grad
        
        # Line search
        step = 1.0
        while step > 1e-10
            alpha_new = alpha + step * delta
            if all(α -> 0.1 < α < 1000.0, alpha_new)
                alpha = alpha_new
                break
            end
            step *= 0.5
        end
        
        if norm(delta) < tol
            return clamp.(alpha, 0.1, 1000.0)
        end
    end
    
    @warn "Newton-Raphson did not converge in $max_iter iterations"
    return clamp.(alpha, 0.1, 1000.0)
end
function estimate_parameters_mle_newton1(Y::Matrix{Float64};
    max_iter::Int=1000,
    tol::Float64=1e-6,
    min_step::Float64=1e-10,
    initial_lambda::Float64=1e-6,
    lambda_increase::Float64=10.0,
    lambda_decrease::Float64=0.1,
    armijo_factor::Float64=1e-4)
   
    n_samples = size(Y, 1)
    n_categories = size(Y, 2)
    
    # Log-likelihood calculation
    function loglikelihood(alpha, Y)
        ll = 0.0
        for i in 1:n_samples
            ll += loggamma(sum(alpha)) - sum(loggamma.(alpha))
            for j in 1:n_categories
                ll += (alpha[j] - 1) * log(Y[i,j])
            end
        end
        return ll
    end
   
    function gradient(alpha)
        grad = zeros(n_categories)
        digamma_sum = digamma(sum(alpha))
       
        for j in 1:n_categories
            grad[j] = n_samples * (digamma_sum - digamma(alpha[j]))
            for i in 1:n_samples
                grad[j] += log(Y[i,j])
            end
        end
       
        return grad
    end
   
    function hessian(alpha)
        H = zeros(n_categories, n_categories)
        trigamma_sum = trigamma(sum(alpha))
       
        for i in 1:n_categories
            for j in 1:n_categories
                if i == j
                    H[i,j] = n_samples * (-trigamma(alpha[i]) + trigamma_sum)
                else
                    H[i,j] = n_samples * trigamma_sum
                end
            end
        end
       
        return H
    end
    
    # Armijo line search condition
    function armijo_condition(alpha, delta, grad, step, ll_current)
        alpha_new = alpha + step * delta
        if any(α -> α ≤ 0.1 || α ≥ 1000.0, alpha_new)
            return false
        end
        ll_new = loglikelihood(alpha_new, Y)
        return ll_new ≥ ll_current + armijo_factor * step * dot(grad, delta)
    end
   
    # Initial guess using MoM
    alpha = estimate_parameters_mom(Y)
    lambda = initial_lambda
    
    # Track progress
    prev_ll = -Inf
    current_ll = loglikelihood(alpha, Y)
    
    for iter in 1:max_iter
        grad = gradient(alpha)
        H = hessian(alpha)
        
        # Early stopping if gradient is nearly zero
        if norm(grad) < tol
            break
        end
        
        # Levenberg-Marquardt style adaptive regularization
        success = false
        while !success && lambda < 1e6
            try
                # Compute update with current regularization
                H_reg = H + lambda * I
                delta = -H_reg \ grad
                
                # Line search with Armijo condition
                step = 1.0
                while step > min_step
                    if armijo_condition(alpha, delta, grad, step, current_ll)
                        alpha_new = alpha + step * delta
                        new_ll = loglikelihood(alpha_new, Y)
                        
                        # Accept update if likelihood improved
                        if new_ll > current_ll
                            alpha = alpha_new
                            prev_ll = current_ll
                            current_ll = new_ll
                            lambda *= lambda_decrease  # Decrease regularization
                            success = true
                            break
                        end
                    end
                    step *= 0.5
                end
            catch e
                # If matrix inversion fails or other numerical issues
                if isa(e, LinearAlgebra.SingularException) || isa(e, LinearAlgebra.PosDefException)
                    lambda *= lambda_increase
                    continue
                else
                    rethrow(e)
                end
            end
            
            if !success
                lambda *= lambda_increase
            end
        end
        
        # Check convergence on likelihood
        if abs(current_ll - prev_ll) < tol * (abs(current_ll) + 1.0)
            break
        end
        
        # Check if we're stuck
        if lambda ≥ 1e6
            @warn "Optimization stuck: regularization too high"
            break
        end
    end
   
    return clamp.(alpha, 0.1, 1000.0)
end

# Custom hybrid method
# function hybrid_optimization(Y::Matrix{Float64})
#     if size(Y, 1) > 20
#         return estimate_parameters_mom(Y)
#     else
#         return estimate_parameters_mle_newton(Y)
#     end
# end

# Hybrid optimization method with q parameter
function hybrid_optimization(Y::Matrix{Float64}, q::Int)
    if size(Y, 1) > q
        return estimate_parameters_mom(Y)
    else
        return estimate_parameters_mle_bobyqa(Y)
    end
end

# Modified log-likelihood calculation
function dirichlet_loglik(y::Vector{Float64}, alpha::Vector{Float64})
    if any(α -> α <= 0, alpha) || any(x -> x <= 0 || x >= 1, y)
        return -Inf
    end
    loglik = loggamma(sum(alpha)) - sum(loggamma.(alpha)) + sum((alpha .- 1) .* log.(y))
    return loglik
end


# Generate example data and run comparison
function generate_compositional_data(n_samples::Int, n_features::Int, n_categories::Int)
    # Generate features
    X = randn(n_samples, n_features)
    
    # Generate compositional responses
    Y = zeros(n_samples, n_categories)
    
    for i in 1:n_samples
        # Create different mixing based on features
        alpha = exp.(0.5 .* X[i, 1:min(n_features, n_categories)])
        
        # Ensure minimum concentration
        alpha = max.(alpha, 0.1)
        
        # Generate Dirichlet random variables
        Y[i, :] = rand(Dirichlet(alpha))
    end
    
    return X, Y
end
# train_csv_path = "C:\\Users\\29827094\\Documents\\GitHub\\DirichletRandomForest\\data_cleaning\\train_data_simulation.csv"
# train_data = CSV.read(train_csv_path, DataFrame)

# # Extract features and responses
# response_cols = names(train_data)[1:3]

# Y_train = Matrix(train_data[:, response_cols])

# Normalize responses
# Y_train = Y_train ./ sum(Y_train, dims=2)
# results = compare_estimation_methods(Y_train)
# print_comparison_results(results)