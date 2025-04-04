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
        
        # Visualization
        p1 = bar(target_dist.fetal_health, target_dist.count, 
                xlabel="Fetal Health", ylabel="Count", 
                title="Target Distribution", legend=false)
        
        correlation_matrix = cor(Matrix(df[:, Not(:fetal_health)]))
        p2 = heatmap(names(df[:, Not(:fetal_health)]), 
                    names(df[:, Not(:fetal_health)]), 
                    correlation_matrix, 
                    title="Feature Correlation", color=:viridis)
        p3 = histogram(df[!,"baseline value"], 
                      xlabel="Baseline Value", ylabel="Count", 
                      title="Baseline Value Distribution", legend=false)
        p4 = histogram(df[!,"accelerations"], 
                      xlabel="Accelerations", ylabel="Count", 
                      title="Accelerations Distribution", legend=false)
       
        plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 800))
    end
