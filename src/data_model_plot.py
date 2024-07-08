import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src.linreg_predict import LinregPredictor


class DataPlotter:
	# Initialize the plotter with paths
	def __init__(self,
			data_path="data/data.csv",
			model_path="data/model.csv",
			normalization_path="data/normalization.csv",
			plot_path="plots/data_plot.png",
			epsilon=1e-8
		):
		self.data_path = data_path
		self.plot_path = plot_path
		self.abort = False
		if self.validate_paths() or self.load_data() or self.validate_data() or self.extract_data():
			self.abort = True
			return
		self.predictor = LinregPredictor(model_path, normalization_path, epsilon)
		if self.predictor.abort:
			self.abort = True
			return
		self.predictor.print_model()

	# validate paths have correct extension and input exist
	def validate_paths(self):
		if not self.data_path.endswith(".csv") or not os.path.exists(self.data_path):
			print("Invalid data file\n")
			sys.exit(1)
		if not self.plot_path.endswith(".png") or not os.path.exists(os.path.dirname(self.plot_path)):
			print("Invalid output file\n")
			sys.exit(1)
	
	# Load data from CSV files
	def load_data(self):
		self.data = pd.read_csv(self.data_path)
		print("\nData file:", self.data_path)
		print(self.data.describe())
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
	
	# Print parameters options for the class
	@staticmethod
	def print_params():
		print(f"""
 ____________________________________________________________________
| choose parameters to launch the plotter:
|     data=PATH              (default = data/data.csv)
|     model=PATH             (default = data/model.csv)
|     norm=PATH              (default = data/normalization.csv)
|     plot=PATH              (default = plots/training_plot.png)
|     epsilon=FLOAT          (default = 1e-8)
|
| you can choose to define any parameter or use the default values
| by pressing enter
|
| example: data=/mydata.csv epsilon=1e-8
 ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
		""")
		
	# Ask user to input parameters and instantiate the class
	@classmethod
	def input_instanciate(self):
		while True:
			try:
				self.print_params()
				params = input("Enter parameters: ")
				params = params.split()
				params = {param.split("=")[0]: param.split("=")[1] for param in params}
				for param in params:
					if param not in ["data", "model", "norm", "plot", "epsilon"]:
						raise ValueError
				break
			except:
				print("Invalid input. Please enter the parameters in the correct format.")

		return DataPlotter(
			data_path=params.get("data", "data/data.csv"),
			model_path=params.get("model", "data/model.csv"),
			normalization_path=params.get("norm", "data/normalization.csv"),
			plot_path=params.get("plot", "plots/data_plot.png"),
			epsilon=float(params.get("epsilon", 1e-8))
		)

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
	predictor = DataPlotter(
		data_path=args.data,
		model_path=args.model,
		normalization_path=args.norm,
		plot_path=args.plot,
		epsilon=args.epsilon
	)
	# predictor = DataPlotter.input_instanciate()
	if not predictor.abort:
		predictor.run()
