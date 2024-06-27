using CSV
using DataFrames
using Plots
using Statistics

plot_path = "plots/model_training_plot.png"
model_path = "model.csv"
normalization_path = "normalization.csv"

epochs = 1000
learning_rate = 0.01
tolerance = 1e-9

# Check if input is valid
if length(ARGS) != 1
    println("Invalid number of arguments")
    println("Usage: julia linreg_train.jl <path to data csv file>")
    exit(1)
elseif !isfile(ARGS[1]) || !occursin(".csv", ARGS[1])
    println("Invalid file")
    println("Usage: julia linreg_train.jl <path to data csv file>")
    exit(1)
end

# Load data and model
data = CSV.read(ARGS[1], DataFrame)

# Check that data has 2 columns and at least 1 line
if size(data)[2] != 2 || size(data)[1] < 1
    println("Data file must have 2 columns and at least 1 line")
    exit(1)
end

# Display extract of data
println("Data file: ", ARGS[1])
println(data[1:min(10, size(data, 1)), :])
println("")

# Separate features and labels
X = data[:, 1] # km
y = data[:, 2] # price

# Normalize the data
X_mean = mean(X)
X_std = std(X)
y_mean = mean(y)
y_std = std(y)

X = (X .- X_mean) ./ X_std
y = (y .- y_mean) ./ y_std

# Initialize parameters
θ0 = 0.0
θ1 = 0.0

# Define the cost function
function compute_cost(X, y, θ0, θ1)
    m = length(y)
    predictions = θ0 .+ θ1 .* X
    sq_errors = (predictions .- y).^2
    return sum(sq_errors) / (2 * m)
end

# Training function
function train(X, y, epochs, learning_rate, tolerance)
    m = length(y)
    θ0 = 0.0
    θ1 = 0.0
    cost_history = zeros(epochs)
    for epoch in 1:epochs
        predictions = θ0 .+ θ1 .* X
        errors = predictions .- y
        θ0_gradient = sum(errors) / m
        θ1_gradient = sum(errors .* X) / m
        θ0 -= learning_rate * θ0_gradient
        θ1 -= learning_rate * θ1_gradient
        cost_history[epoch] = compute_cost(X, y, θ0, θ1)
        println("Epoch $epoch: Cost = $(cost_history[epoch])")
        
        # Check for convergence
        # if epoch > 1 && abs(cost_history[epoch] - cost_history[epoch - 1]) < tolerance
        #     println("Convergence reached at epoch $epoch")
        #     cost_history = cost_history[1:epoch]
        #     break
        # end
    end
    return θ0, θ1, cost_history
end

# Train the model
θ0, θ1, cost_history = train(X, y, epochs, learning_rate, tolerance)

# Save the model parameters
model = DataFrame(θ0 = [θ0], θ1 = [θ1])
CSV.write(model_path, model)

# Save the normalization parameters
normalization_params = DataFrame(parameter=["X_mean", "X_std", "y_mean", "y_std"], value=[X_mean, X_std, y_mean, y_std])
CSV.write(normalization_path, normalization_params)

# Plot the cost history
plot(1:length(cost_history), cost_history, xlabel="Epochs", ylabel="Cost", title="Cost vs Epochs", legend=false)
savefig(plot_path)

println("Training complete.")
println("Model parameters: θ0 = $θ0, θ1 = $θ1")
println("Cost history plot saved to $plot_path")
println("Model parameters saved to $model_path")
println("Normalization parameters saved to $normalization_path")
