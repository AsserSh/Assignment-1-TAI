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


# Exploratory Data Analysis (EDA)
    function explore_data(df)
        println("Data dimensions: ", size(df))
        println("\nSample records:")
        println(first(df, 5))
    
        println("\nStatistics summary:")
        println(describe(df))
        
        println("\nMissing values per column:")
        println(combine(df, names(df) .=> (x -> sum(ismissing.(x))) .=> names(df)))
        
        # Distribution of target variable
        target_dist = combine(groupby(df, :fetal_health), nrow => :count)
        println("\nTarget variable distribution:")
        println(target_dist)
        

