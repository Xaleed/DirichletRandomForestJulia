module DirichletRandomForest

# Export functions and types that users should have access to
export DirichletForest, fit_dirichlet_forest!, predict_dirichlet_forest,
       aitchison_distance, compositional_r2

# Include other files in the src directory
include("MLE_vs_MoM.jl")       # Parameter estimation methods
include("dirichlet_forest.jl") # Main implementation
include("evaluation.jl")       # Evaluation metrics

end