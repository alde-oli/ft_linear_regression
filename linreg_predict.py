import os
import sys
import pandas as pd
import argparse


class LinregPredictor:
	# Initialize the predictor with paths
	def __init__(self, model_path="data/model.csv", normalization_path="data/normalization.csv", epsilon=1e-8):
		self.model_path = model_path
		self.normalization_path = normalization_path
		self.epsilon = epsilon
		self.validate_paths()
		self.load_data()
		self.validate_data()
		self.extract_data()

	# Validate paths have correct extension and input exist
	def validate_paths(self):
		if not self.model_path.endswith(".csv") or not os.path.exists(self.model_path):
			print("Invalid model file\n")
			sys.exit(1)
		if not self.normalization_path.endswith(".csv") or not os.path.exists(self.normalization_path):
			print("Invalid normalization file\n")
			sys.exit(1)

	# Load data from CSV files
	def load_data(self):
		self.model = pd.read_csv(self.model_path)
		self.normalization = pd.read_csv(self.normalization_path)

	# Validate the loaded files to ensure they have the correct format
	def validate_data(self):
		if self.model.shape != (1, 2) or not all(self.model.columns == ["theta0", "theta1"]):
			print("Model file must have 2 values, theta1 and theta0")
			sys.exit(1)
		if self.normalization.shape != (4, 2) or not all(self.normalization.parameter == ["X_mean", "X_std", "y_mean", "y_std"]):
			print("Normalization file must have 4 values: X_mean, X_std, y_mean, y_std")
			sys.exit(1)

	# Extract model and normalization parameters from the loaded files
	def extract_data(self):
		print("Model file:", self.model_path)
		print(self.model)
		print("")
		print("Normalization file:", self.normalization_path)
		print(self.normalization)
		print("")
		self.theta1 = self.model["theta1"][0]
		self.theta0 = self.model["theta0"][0]
		self.X_mean = self.normalization.loc[self.normalization["parameter"] == "X_mean", "value"].values[0]
		self.X_std = self.normalization.loc[self.normalization["parameter"] == "X_std", "value"].values[0]
		self.y_mean = self.normalization.loc[self.normalization["parameter"] == "y_mean", "value"].values[0]
		self.y_std = self.normalization.loc[self.normalization["parameter"] == "y_std", "value"].values[0]
		if self.X_std + self.epsilon == 0 or self.y_std + self.epsilon == 0:
			print("Cannot normalize data, standard deviation is zero")
			sys.exit(1)

	# Predict the price based on the number of kilometers using the model parameters and normalization parameters
	def predict(self, km):
		if self.X_std or self.X_mean:
			km = (km - self.X_mean) / (self.X_std + self.epsilon)
		price = self.theta0 + self.theta1 * km
		if self.y_std or self.y_mean:
			price = price * self.y_std + self.y_mean
		return price

	# Prompt user for input, make prediction, and print the predicted price
	def run(self):
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


# Parse command line arguments
def parse_args():
	parser = argparse.ArgumentParser(description='Run linear regression predictions.')
	parser.add_argument('--model', type=str, help='Path to the model CSV file', default="data/model.csv")
	parser.add_argument('--norm', type=str, help='Path to the normalization CSV file', default="data/normalization.csv")
	parser.add_argument('--epsilon', type=float, help='Small value to prevent division by zero', default=1e-8)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	predictor = LinregPredictor(model_path=args.model, normalization_path=args.norm, epsilon=args.epsilon)
	predictor.run()
