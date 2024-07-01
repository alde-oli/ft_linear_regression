using CSV
using DataFrames
using Plots
using Statistics

plot_path = "plots/model_training_plot.png"
model_path = "model.csv"
normalization_path = "normalization.csv"

epochs = 500
learning_rate = 0.01
tolerance = 0.20


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
weight = 0.0
bias = 0.0


# Define the cost function
function compute_cost(X, y, weight, bias)
    m = length(y)
    predictions = weight .+ bias .* X
    sq_errors = (predictions .- y).^2
    return sum(sq_errors) / (2 * m)
end


# Define the accuracy function with tolerance
function compute_accuracy(X, y, weight, bias, tolerance)
    predictions = weight .+ bias .* X
    abs_errors = abs.(predictions .- y)
    return mean(abs_errors .< tolerance)
end


# Training function
function train(X, y, epochs, learning_rate, tolerance)
    m = length(y)
    weight = 0.0
    bias = 0.0
    recorded_costs = zeros(epochs)
    recorded_accuracies = zeros(epochs)
    for epoch in 1:epochs
        predictions = weight .+ bias .* X
        errors = predictions .- y
        weight_gradient = sum(errors) / m
        bias_gradient = sum(errors .* X) / m
        weight -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient
        recorded_costs[epoch] = compute_cost(X, y, weight, bias)
        recorded_accuracies[epoch] = compute_accuracy(X, y, weight, bias, tolerance)
    end
    return weight, bias, recorded_costs, recorded_accuracies
end


# Train the model
weight, bias, recorded_costs, recorded_accuracies = train(X, y, epochs, learning_rate, tolerance)


# Save the model and normalization parameters
model = DataFrame(weight = [weight], bias = [bias])
CSV.write(model_path, model)
normalization_params = DataFrame(parameter=["X_mean", "X_std", "y_mean", "y_std"], value=[X_mean, X_std, y_mean, y_std])
CSV.write(normalization_path, normalization_params)


# Plot the cost and tolerance-based accuracy
plot(1:epochs, recorded_costs, label="Cost (MSE)", xlabel="Epoch", ylabel="Metric", title="Model Training", lw=2)
plot!(1:epochs, recorded_accuracies, label="Tolerance Accuracy", lw=2)
println("Model trained and saved")
savefig(plot_path)
