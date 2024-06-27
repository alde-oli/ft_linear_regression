using CSV
using DataFrames

model_path = ARGS[1]
normalization_path = "normalization.csv"

# Check if model file is valid
if length(ARGS) != 1 || !isfile(model_path)
    println("Invalid number of arguments or file not found")
    println("Usage: julia linreg_predict.jl <path to model csv file>")
    exit(1)
end

# Load model parameters
model = CSV.read(model_path, DataFrame)
θ0 = model.θ0[1]
θ1 = model.θ1[1]

# Load normalization parameters
normalization_params = CSV.read(normalization_path, DataFrame)
X_mean = normalization_params.value[1]
X_std = normalization_params.value[2]
y_mean = normalization_params.value[3]
y_std = normalization_params.value[4]

# Function to make predictions
function predict(km, θ0, θ1, X_mean, X_std, y_mean, y_std)
    km_normalized = (km - X_mean) / X_std
    price_normalized = θ0 + θ1 * km_normalized
    price = price_normalized * y_std + y_mean
    return price
end

println("Enter the value for the feature (km):")
km = parse(Float64, readline())
predicted_price = predict(km, θ0, θ1, X_mean, X_std, y_mean, y_std)
println("The predicted price is: $predicted_price")
