using CSV
using DataFrames
using Plots

# Path to save the plot
plot_path = "plots/data_plot.png"

# Check if input is valid
if length(ARGS) != 3
    println("Invalid number of arguments")
    println("Usage: julia data_plot.jl <path to data csv file> <path to model csv file> <path to normalization csv file>")
    exit(1)
elseif !isfile(ARGS[1]) || !isfile(ARGS[2]) || !isfile(ARGS[3]) || ARGS[1] == ARGS[2] || ARGS[1] == ARGS[3] || ARGS[2] == ARGS[3] || !occursin(".csv", ARGS[1]) || !occursin(".csv", ARGS[2]) || !occursin(".csv", ARGS[3])
    println("Invalid file")
    println("Usage: julia data_plot.jl <path to data csv file> <path to model csv file> <path to normalization csv file>")
    exit(1)
end

# Load data, model, and normalization parameters
data = CSV.read(ARGS[1], DataFrame)
model = CSV.read(ARGS[2], DataFrame)
normalization_params = CSV.read(ARGS[3], DataFrame)

# Check that model has 2 columns and 1 line (+ header)
if size(model)[2] != 2 || size(model)[1] != 1
    println("Model file must have 2 values, weight and bias")
    exit(1)
end

# Check that data has 2 columns
if size(data)[2] != 2 || size(data)[1] < 1
    println("Data file must have 2 columns and at least 1 line")
    exit(1)
end

# Extract normalization parameters
X_mean = normalization_params.value[1]
X_std = normalization_params.value[2]
y_mean = normalization_params.value[3]
y_std = normalization_params.value[4]

# Display data and model
println("Data file: ", ARGS[1])
println(data)

println("\nModel file: ", ARGS[2])
println(model)

println("\nNormalization file: ", ARGS[3])
println(normalization_params)

# Check if output file is valid
if !occursin(".png", plot_path)
    println("Invalid output file")
    exit(1)
end

# Normalize the data
X = (data[:, 1] .- X_mean) ./ X_std
y = (data[:, 2] .- y_mean) ./ y_std

# Extract model parameters
θ0 = model.θ0[1]
θ1 = model.θ1[1]

# Generate predictions
predictions_normalized = θ0 .+ θ1 .* X
predictions = predictions_normalized .* y_std .+ y_mean

# Plot data and model using model's column names as labels
plot(data[:, 1], data[:, 2], label="Data", seriestype=:scatter)
plot!(data[:, 1], predictions, label="Model", xlabel=names(data)[1], ylabel=names(data)[2])
savefig(plot_path)
