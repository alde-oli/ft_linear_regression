import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LinregTrainer:
	# Initialize the trainer with paths and hyperparameters
	def __init__(self,
			data_path="data/data.csv",
			model_path="data/model.csv",
			normalization_path="data/normalization.csv",
			epochs=500,
			learning_rate=0.01,
			epsilon=1e-8,
			plot_path="plots/training_plot.png"
		):
		self.data_path = data_path
		self.plot_path = plot_path
		self.model_path = model_path
		self.normalization_path = normalization_path
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.epsilon = epsilon
		self.abort = False
		if self.validate_paths() or self.load_data() or self.validate_files() or self.normalize_data():
			self.abort = True
			return

	# Validate paths before loading files
	def validate_paths(self):
		if not self.data_path.endswith(".csv") or not os.path.exists(self.data_path):
			print("Invalid file")
			return -1
		if not self.model_path.endswith(".csv") or not os.path.exists(os.path.dirname(self.model_path)):
			print("Invalid model file")
			return -1
		if not self.normalization_path.endswith(".csv") or not os.path.exists(os.path.dirname(self.normalization_path)):
			print("Invalid normalization file")
			return -1
		if not self.plot_path.endswith(".png") or not os.path.exists(os.path.dirname(self.plot_path)):
			print("Invalid plot file")
			return -1
		return 0

	# Load the data from the CSV file and split features and target
	def load_data(self):
		self.data = pd.read_csv(self.data_path)
		if self.data.shape[1] != 2 or self.data.shape[0] < 1:
			print("Data file must have 2 columns and at least 1 line")
			return -1
		print("\nData file:", self.data_path)
		print(self.data.describe())
		print("")
		self.X = self.data.iloc[:, 0].values
		self.y = self.data.iloc[:, 1].values
		return 0
	
	# Validate the loaded files to ensure they have the correct format
	def validate_files(self):
		if self.data.shape[1] != 2 or self.data.shape[0] < 1:
			print("Data file must have 2 columns and at least 1 line")
			return -1
		return 0

	# Normalize the features and target values
	def normalize_data(self):
		self.X_mean = self.X.mean()
		self.X_std = self.X.std()
		self.y_mean = self.y.mean()
		self.y_std = self.y.std()
		if self.X_std + self.epsilon == 0 or self.y_std + self.epsilon == 0:
			print("Cannot normalize data, standard deviation is zero")
			return -1
		self.X = (self.X - self.X_mean) / (self.X_std + self.epsilon)
		self.y = (self.y - self.y_mean) / (self.y_std + self.epsilon)

		normalization_params = pd.DataFrame({
			"parameter": ["X_mean", "X_std", "y_mean", "y_std"],
			"value": [self.X_mean, self.X_std, self.y_mean, self.y_std]
		})
		normalization_params.to_csv(self.normalization_path, index=False)
		return 0


	# Train the linear regression model using gradient descent
	def train(self):
		m = len(self.y)
		theta0 = 0.0
		theta1 = 0.0
		recorded_errors = np.zeros(self.epochs)

		for epoch in range(self.epochs):
			predictions = theta0 + theta1 * self.X
			errors = predictions - self.y
			# Calculate gradients
			tmp_theta0 = self.learning_rate * np.sum(errors) / m
			tmp_theta1 = self.learning_rate * np.sum(errors * self.X) / m
			# Update parameters simultaneously
			theta0 -= tmp_theta0
			theta1 -= tmp_theta1
			# Record cost and accuracy for this epoch
			recorded_errors[epoch] = np.sum(abs(errors)) / m

		self.theta0 = theta0
		self.theta1 = theta1
		self.recorded_errors = recorded_errors

	# Save the trained model parameters to a CSV file
	def save_model(self):
		model = pd.DataFrame({"theta0": [self.theta0], "theta1": [self.theta1]})
		model.to_csv(self.model_path, index=False)

	# Plot the cost and accuracy over epochs
	def plot_training(self):
		plt.plot(range(1, self.epochs + 1), self.recorded_errors, label="Error", lw=2)
		plt.xlabel("Epoch")
		plt.ylabel("Error")
		plt.title("Model Training")
		plt.legend()
		plt.savefig(self.plot_path)
	
	#print parameters options for the class
	@staticmethod
	def print_params():
		print(f"""
 ____________________________________________________________________
| choose parameters to launch the trainer:
|     data=PATH              (default = data/data.csv)
|     model=PATH             (default = data/model.csv)
|     norm=PATH              (default = data/normalization.csv)
|     epochs=INT             (default = 500)
|     learning_rate=FLOAT    (default = 0.01)
|     epsilon=FLOAT          (default = 1e-8)
|     plot=PATH              (default = plots/training_plot.png)
|
| you can choose to define any parameter or use the default values
| by pressing enter
|
| example: data=/mydata.csv epsilon=1e-8 epochs=1000
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
					if param not in ["data", "model", "norm", "epochs", "learning_rate", "epsilon", "plot"]:
						raise ValueError
				break
			except:
				print("Invalid input. Please enter the parameters in the correct format.")

		return LinregTrainer(
			data_path=params.get("data", "data/data.csv"),
			model_path=params.get("model", "data/model.csv"),
			normalization_path=params.get("norm", "data/normalization.csv"),
			epochs=int(params.get("epochs", 500)),
			learning_rate=float(params.get("learning_rate", 0.01)),
			epsilon=float(params.get("epsilon", 1e-8)),
			plot_path=params.get("plot", "plots/training_plot.png")
		)

	#print model, normalization and last error
	def print_model(self):
		x_mean = f"{self.X_mean:<20.8f}"
		x_std = f"{self.X_std:<20.8f}"
		y_mean = f"{self.y_mean:<20.8f}"
		y_std = f"{self.y_std:<20.8f}"
		last_error = f"{self.recorded_errors[-1]:<20.8f}"

		print(f"""
 ____________________________________________________________________
| Model:
|     func: {self.theta1} * X + {self.theta0}
|
|     Normalization:
|         x_mean: {x_mean} x_std: {x_std}
|         y_mean: {y_mean} y_std: {y_std}
|
|     Error: {last_error}
|     Training time: {self.training_time} seconds
|
|     Model saved to: {self.model_path}
|     Normalization saved to: {self.normalization_path}
|     Error plot saved to: {self.plot_path}
 ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
		""")

	# Execute the training process and save the results.
	def run(self):
		start_time = time.time()
		self.train()
		self.training_time = time.time() - start_time
		self.save_model()
		self.print_model()
		self.plot_training()
		print("Model trained and saved")
# LinregTrainer


# Parse command line arguments
def parse_args():
	parser = argparse.ArgumentParser(description='Run linear regression training.')
	parser.add_argument(
		'--data',
		type=str,
		help='Path to the data CSV file',
		default="data/data.csv"
		)
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
		'--epochs',
		type=int,
		help='Number of training epochs',
		default=500
	)
	parser.add_argument(
		'--learning_rate',
		type=float,
		help='Gradient descent learning rate',
		default=0.01
	)
	parser.add_argument(
		'--epsilon',
		type=float, 
		help='Small value to prevent division by zero',
		default=1e-8
	)
	parser.add_argument(
		'--plot',
		type=str,
		help='Path to save the training plot',
		default="plots/training_plot.png"
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	trainer = LinregTrainer(
		data_path=args.data,
		model_path=args.model,
		normalization_path=args.norm,
		epochs=args.epochs,
		learning_rate=args.learning_rate,
		epsilon=args.epsilon,
		plot_path=args.plot
	)
	# trainer = LinregTrainer.input_instanciate()
	if not trainer.abort:
		trainer.run()
