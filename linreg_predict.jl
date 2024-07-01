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


# Load model parametersjul
model = CSV.read(model_path, DataFrame)
weight = model.weight[1]
bias = model.bias[1]


# Load normalization parameters
normalization_params = CSV.read(normalization_path, DataFrame)
X_mean = normalization_params.value[1]
X_std = normalization_params.value[2]
y_mean = normalization_params.value[3]
y_std = normalization_params.value[4]


# Function to make predictions
function predict(km, weight, bias, X_mean, X_std, y_mean, y_std)
    km_normalized = (km - X_mean) / X_std
    price_normalized = weight + bias * km_normalized
    price = price_normalized * y_std + y_mean
    return price
end


# ask for input until valid number
km = 0.
while true
    print("Enter the number of kilometers: ")
    global km = parse(Float64, readline())
    if km >= 0
        break
    else
        println("Invalid number of kilometers")
    end
end
predicted_price = predict(km, weight, bias, X_mean, X_std, y_mean, y_std)
println("The predicted price is: $predicted_price")
