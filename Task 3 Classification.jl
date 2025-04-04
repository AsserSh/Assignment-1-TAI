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
# Data Preprocessing
function prepare_data(df)
    df.fetal_health = categorical(df.fetal_health)
    y = df.fetal_health
    X = select(df, Not(:fetal_health))
    
    normalizer = Standardizer()
    model = machine(normalizer, X)
    fit!(model)
    X = MLJ.transform(model, X)
    
    return X, y
end

X, y = prepare_data(df)

# Handling class imbalance
function balance_classes(X, y)
    class_counts = combine(groupby(DataFrame(y=y), :y), nrow => :count)
    println("\nClass distribution before balancing:")
    println(class_counts)
    
    X_matrix = Float32.(Matrix(X))
    y_vector = Int.(y.refs)
    
    X_resampled, y_resampled = smote(X_matrix, y_vector; k=5, ratios=Dict(1=>1.0, 2=>1.0, 3=>1.0))
    
    X_final = DataFrame(X_resampled, names(X))
    y_final = categorical(y_resampled)
    
    println("\nClass distribution after balancing:")
    println(combine(groupby(DataFrame(y=y_final), :y), nrow => :count))
    
    return X_final, y_final
end
