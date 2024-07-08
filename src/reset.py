import os
import sys
import argparse
import pandas as pd

# Create clean .csv files and remove .png files
def reset_files(
		model_path="data/model.csv",
		normalization_path="data/normalization.csv",
		data_plot_path="plots/data_plot.png",
		training_plot_path="plots/training_plot.png"
	):
	if not model_path.endswith(".csv") or not os.path.exists(os.path.dirname(model_path)):
		print("Invalid model file\n")
		return -1
	else:
		model = pd.DataFrame({"theta0": [0], "theta1": [0]})
		model.to_csv(model_path, index=False)
	if not normalization_path.endswith(".csv") or not os.path.exists(os.path.dirname(normalization_path)):
		print("Invalid normalization file\n")
		return -1
	else:
		normalization = pd.DataFrame({
			"parameter": ["X_mean", "X_std", "y_mean", "y_std"],
			"value": [0, 1, 0, 1]
		})
		normalization.to_csv(normalization_path, index=False)
	if not data_plot_path.endswith(".png"):
		print("Invalid data plot file\n")
		return -1
	else:
		if os.path.exists(data_plot_path):
			os.remove(data_plot_path)
	if not training_plot_path.endswith(".png"):
		print("Invalid training plot file\n")
		return -1
	else:
		if os.path.exists(training_plot_path):
			os.remove(training_plot_path)
	return 0


# Print parameters to use for the cleaner
def print_cleaner_params():
	print(f"""
 ____________________________________________________________________
| choose parameters to launch the cleaner:
|     model=PATH             (default = data/model.csv)
|     norm=PATH              (default = data/normalization.csv)
|     data_plot=PATH         (default = plots/data_plot.png)
|     training_plot=PATH     (default = plots/training_plot.png)
|
| you can choose to define any parameter or use the default values
| by pressing enter
|
| example: norm=data/normalization.csv epsilon=1e-9
 ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
	""")
	

# Ask user to input parameters and launch the cleaner
def input_reset_files():
	while True:
		try:
			print_cleaner_params()
			params = input("Enter parameters: ")
			params = params.split()
			params = {param.split("=")[0]: param.split("=")[1] for param in params}
			for param in params:
				if param not in ["model", "norm", "data_plot", "training_plot"]:
					raise ValueError
			break
		except:
			print("Invalid input. Please enter the parameters in the correct format.")

	return reset_files(
		model_path=params.get("model", "data/model.csv"),
		normalization_path=params.get("norm", "data/normalization.csv"),
		data_plot_path=params.get("data_plot", "plots/data_plot.png"),
		training_plot_path=params.get("training_plot", "plots/training_plot.png")
	)


# Parse command line arguments
def parse_args():
	parser = argparse.ArgumentParser(description='Run linear regression training.')
	parser.add_argument(
		'--model',
		type=str,
		help='Path to save the model CSV file',
		default="data/model.csv"
	)
	parser.add_argument(
		'--norm',
		type=str,
		help='Path to save the normalization CSV file',
		default="data/normalization.csv"
	)
	parser.add_argument(
		'--data_plot',
		type=str,
		help='Path to save the training plot',
		default="plots/data_plot.png"
	)
	parser.add_argument(
		'--training_plot',
		type=str,
		help='Path to save the model plot',
		default="plots/training_plot.png"
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	reset_files(
		model_path=args.model,
		normalization_path=args.norm,
		data_plot_path=args.data_plot,
		training_plot_path=args.training_plot
	)