import os
import sys
import pandas as pd
import argparse


class LinregPredictor:
	# Initialize the predictor with paths
	def __init__(self,
			model_path="data/model.csv",
			normalization_path="data/normalization.csv",
			epsilon=1e-8
		):
		self.model_path = model_path
		self.normalization_path = normalization_path
		self.epsilon = epsilon
		self.abort = False
		if self.validate_paths() or self.load_data() or self.validate_data() or self.extract_data():
			self.abort = True
			return

	# Validate paths have correct extension and input exist
	def validate_paths(self):
		if not self.model_path.endswith(".csv") or not os.path.exists(self.model_path):
			print("Invalid model file\n")
			return -1
		if not self.normalization_path.endswith(".csv") or not os.path.exists(self.normalization_path):
			print("Invalid normalization file\n")
			return -1
		return 0

	# Load data from CSV files
	def load_data(self):
		self.model = pd.read_csv(self.model_path)
		self.normalization = pd.read_csv(self.normalization_path)

	# Validate the loaded files to ensure they have the correct format
	def validate_data(self):
		if self.model.shape != (1, 2) or not all(self.model.columns == ["theta0", "theta1"]):
			print("Model file must have 2 values, theta1 and theta0")
			return -1
		if self.normalization.shape != (4, 2) or not all(self.normalization.parameter == ["X_mean", "X_std", "y_mean", "y_std"]):
			print("Normalization file must have 4 values: X_mean, X_std, y_mean, y_std")
			return -1
		return 0

	# Extract model and normalization parameters from the loaded files
	def extract_data(self):
		self.theta1 = self.model["theta1"][0]
		self.theta0 = self.model["theta0"][0]
		self.X_mean = self.normalization.loc[self.normalization["parameter"] == "X_mean", "value"].values[0]
		self.X_std = self.normalization.loc[self.normalization["parameter"] == "X_std", "value"].values[0]
		self.y_mean = self.normalization.loc[self.normalization["parameter"] == "y_mean", "value"].values[0]
		self.y_std = self.normalization.loc[self.normalization["parameter"] == "y_std", "value"].values[0]
		if self.X_std + self.epsilon == 0 or self.y_std + self.epsilon == 0:
			print("Cannot normalize data, standard deviation is zero")
			return -1
		return 0

	# Predict the price based on the number of kilometers using the model parameters and normalization parameters
	def predict(self, km):
		if self.X_std or self.X_mean:
			km = (km - self.X_mean) / (self.X_std + self.epsilon)
		price = self.theta0 + self.theta1 * km
		if self.y_std or self.y_mean:
			price = price * self.y_std + self.y_mean
		return price

	# Print the parameters options for the class
	@staticmethod
	def print_params():
		print(f"""
 ____________________________________________________________________
| choose parameters to launch the predictor:
|     model=PATH             (default = data/model.csv)
|     norm=PATH              (default = data/normalization.csv)
|     epsilon=FLOAT          (default = 1e-8)
|
| you can choose to define any parameter or use the default values
| by pressing enter
|
| example: norm=data/normalization.csv epsilon=1e-9
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
					if param not in ["model", "norm", "epsilon"]:
						raise ValueError
				break
			except:
				print("Invalid input. Please enter the parameters in the correct format.")

		return LinregPredictor(
			model_path=params.get("model", "data/model.csv"),
			normalization_path=params.get("norm", "data/normalization.csv"),
			epsilon=float(params.get("epsilon", 1e-8))
		)

	# Print model, normalization and last error
	def print_model(self):
		x_mean = f"{self.X_mean:<20.8f}"
		x_std = f"{self.X_std:<20.8f}"
		y_mean = f"{self.y_mean:<20.8f}"
		y_std = f"{self.y_std:<20.8f}"

		print(f"""
 ____________________________________________________________________
| Model:
|     func: {self.theta1} * X + {self.theta0}
|
|     Normalization:
|         x_mean: {x_mean} x_std: {x_std}
|         y_mean: {y_mean} y_std: {y_std}
 ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
		""")

	# Prompt user for input, make prediction, and print the predicted price
	def run(self):
		self.print_model()

		while True:
			try:
				km = float(input("Enter the number of kilometers: "))
				if km >= 0:
					break
				else:
					print("Invalid number of kilometers")
			except ValueError:
				print("Invalid input. Please enter a number.")

		predicted_price = self.predict(km)
		print(f"The predicted price is: {predicted_price:.2f}")
# LinregPredictor


# Parse command line arguments
def parse_args():
	parser = argparse.ArgumentParser(description='Run linear regression predictions.')
	parser.add_argument(
		'--model',
		type=str,
		help='Path to the model CSV file',
		default="data/model.csv"
	)
	parser.add_argument(
		'--norm',
		type=str,
		help='Path to the normalization CSV file',
		default="data/normalization.csv"
	)
	parser.add_argument(
		'--epsilon',
		type=float,
		help='Small value to prevent division by zero',
		default=1e-8
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	predictor = LinregPredictor(
		model_path=args.model,
		normalization_path=args.norm,
		epsilon=args.epsilon
	)
	# predictor = LinregPredictor.input_instanciate()
	if not predictor.abort:
		predictor.run()
