using Pkg
Pkg.status()
Pkg.add(["DataFrames", "CSV", "GLM", "Statistics", "Plots"])
using DataFrames, CSV, GLM, Statistics, Plots


# Load the dataset
file_path = "C:\\Users\\Asser\\Downloads\\global_food_wastage_dataset.csv"
df = CSV.read(file_path, DataFrame)


# Filter for Dairy Products only
dairy_df = filter(row -> row."Food Category" == "Dairy Products", df)
