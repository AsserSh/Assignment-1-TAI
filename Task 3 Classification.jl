# Load required packages
using Pkg
Pkg.add(["DataFrames", "CSV", "MLJ", "StatsPlots", "Plots", "Random", "Imbalance", "CategoricalArrays", "DecisionTree", "MLJLinearModels", "MLJFlux"])
using DataFrames, CSV, MLJ, StatsPlots, Plots, Random, Imbalance, CategoricalArrays, DecisionTree,MLJLinearModels, MLJFlux

Pkg.add("MLJDecisionTreeInterface")  # For RandomForestClassifier
Pkg.add("XGBoost")                   # For XGBoostClassifier
Pkg.add("MLJLinearModels")           # For LogisticClassifier
Pkg.add("MLJFlux") 
Pkg.add("MLJXGBoostInterface")

# Load the data
data_path = "C:\\Users\\Asser\\Downloads\\fetal_health.csv"
df = CSV.read(data_path, DataFrame)

