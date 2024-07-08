import os
import sys
import argparse
import pandas as pd

#create clean .csv files and remove .png files
def reset_files(model_path, normalization_path, data_plot_path, training_plot_path):
	if not model_path.endswith(".csv"):
		print("Invalid model file\n")
		sys.exit(1)
	else:
		model = pd.DataFrame({"theta0": [0], "theta1": [0]})
		model.to_csv(model_path, index=False)
	if not normalization_path.endswith(".csv"):
		print("Invalid normalization file\n")
		sys.exit(1)
	else:
		normalization = pd.DataFrame({
			"parameter": ["X_mean", "X_std", "y_mean", "y_std"],
			"value": [0, 1, 0, 1]
		})
		normalization.to_csv(normalization_path, index=False)
	if not data_plot_path.endswith(".png"):
		print("Invalid data plot file\n")
		sys.exit(1)
	else:
		if os.path.exists(data_plot_path):
			os.remove(data_plot_path)
	if not training_plot_path.endswith(".png"):
		print("Invalid training plot file\n")
		sys.exit(1)
	else:
		if os.path.exists(training_plot_path):
			os.remove(training_plot_path)


# Parse command line arguments
def parse_args():
	parser = argparse.ArgumentParser(description='Run linear regression training.')
	parser.add_argument('--model', type=str, help='Path to save the model CSV file', default="data/model.csv")
	parser.add_argument('--norm', type=str, help='Path to save the normalization CSV file', default="data/normalization.csv")
	parser.add_argument('--data_plot', type=str, help='Path to save the training plot', default="plots/data_plot.png")
	parser.add_argument('--training_plot', type=str, help='Path to save the model plot', default="plots/training_plot.png")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	reset_files(args.model, args.norm, args.data_plot, args.training_plot)