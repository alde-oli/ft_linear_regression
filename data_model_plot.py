import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from linreg_predict import LinregPredictor


class DataPlotter:
	# Initialize the plotter with paths
	def __init__(self, data_path="data/data.csv", model_path="data/model.csv", normalization_path="data/normalization.csv", plot_path="plots/data_plot.png", epsilon=1e-8):
		self.data_path = data_path
		self.plot_path = plot_path
		self.validate_paths()
		self.load_data()
		self.validate_data()
		self.extract_data()
		self.predictor = LinregPredictor(model_path, normalization_path, epsilon)

	# validate paths have correct extension and input exist
	def validate_paths(self):
		if not self.data_path.endswith(".csv") or not os.path.exists(self.data_path):
			print("Invalid data file\n")
			sys.exit(1)
		if not self.plot_path.endswith(".png"):
			print("Invalid output file\n")
			sys.exit(1)
	
	# Load data from CSV files
	def load_data(self):
		self.data = pd.read_csv(self.data_path)
		print("Data file:", self.data_path)
		print(self.data.head(min(10, self.data.shape[0])))
		print("")
	
	# Validate the loaded files to ensure they have the correct format
	def validate_data(self):
		if self.data.shape[1] != 2 or self.data.shape[0] < 1:
			print("Data file must have 2 columns and at least 1 line")
			sys.exit(1)
	
	# Extract data from the loaded files
	def extract_data(self):
		self.X = self.data.iloc[:, 0]
		self.y = self.data.iloc[:, 1]
	
	# Generate predictions using the LinregPredictor instance
	def predict(self):
		self.predictions = self.X.apply(self.predictor.predict)
	
	# Plot the data and the model predictions, and save the plot to a file
	def plot_data(self):
		plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], label="Data")
		plt.plot(self.data.iloc[:, 0], self.predictions, label="Model", color='red')
		plt.xlabel(self.data.columns[0])
		plt.ylabel(self.data.columns[1])
		plt.legend()
		plt.title("Data and Model")
		plt.savefig(self.plot_path)
		print(f"Plot saved to {self.plot_path}")

	# Run everything
	def run(self):
		self.predict()
		self.plot_data()
# DataPlotter


# Parse command line arguments
def parse_args():
	parser = argparse.ArgumentParser(description='Run data plotter.')
	parser.add_argument('--data', type=str, help='Path to data CSV file', default="data/data.csv")
	parser.add_argument('--model', type=str, help='Path to model CSV file', default="data/model.csv")
	parser.add_argument('--norm', type=str, help='Path to normalization CSV file', default="data/normalization.csv")
	parser.add_argument('--plot', type=str, help='Path to save the plot', default="plots/data_plot.png")
	parser.add_argument('--epsilon', type=float, help='Epsilon value for normalization', default=1e-8)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	predictor = DataPlotter(data_path=args.data, model_path=args.model, normalization_path=args.norm, plot_path=args.plot, epsilon=args.epsilon)
	predictor.run()
