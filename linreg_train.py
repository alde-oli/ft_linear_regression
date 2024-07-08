import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LinregTrainer:
	# Initialize the trainer with paths and hyperparameters
	def __init__(self, data_path="data/data.csv", model_path="data/model.csv", normalization_path="data/normalization.csv", epochs=500, learning_rate=0.01, epsilon=1e-8, plot_path="plots/training_plot.png"):
		self.data_path = data_path
		self.plot_path = plot_path
		self.model_path = model_path
		self.normalization_path = normalization_path
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.epsilon = epsilon
		self.validate_paths()
		self.load_data()
		self.validate_files()
		self.normalize_data()

	# Validate paths before loading files
	def validate_paths(self):
		if not self.data_path.endswith(".csv") or not os.path.exists(self.data_path):
			print("Invalid file")
			sys.exit(1)
		if not self.model_path.endswith(".csv"):
			print("Invalid model file")
			sys.exit(1)
		if not self.normalization_path.endswith(".csv"):
			print("Invalid normalization file")
			sys.exit(1)
		if not self.plot_path.endswith(".png"):
			print("Invalid plot file")
			sys.exit(1)

	# Load the data from the CSV file and split features and target
	def load_data(self):
		self.data = pd.read_csv(self.data_path)
		if self.data.shape[1] != 2 or self.data.shape[0] < 1:
			print("Data file must have 2 columns and at least 1 line")
			sys.exit(1)
		print("Data file:", self.data_path)
		print(self.data.head(min(10, self.data.shape[0])))
		print("")
		self.X = self.data.iloc[:, 0].values
		self.y = self.data.iloc[:, 1].values
	
	# Validate the loaded files to ensure they have the correct format
	def validate_files(self):
		if self.data.shape[1] != 2 or self.data.shape[0] < 1:
			print("Data file must have 2 columns and at least 1 line")
			sys.exit(1)

	# Normalize the features and target values
	def normalize_data(self):
		self.X_mean = self.X.mean()
		self.X_std = self.X.std()
		self.y_mean = self.y.mean()
		self.y_std = self.y.std()
		if self.X_std + self.epsilon == 0 or self.y_std + self.epsilon == 0:
			print("Cannot normalize data, standard deviation is zero")
			sys.exit(1)
		self.X = (self.X - self.X_mean) / (self.X_std + self.epsilon)
		self.y = (self.y - self.y_mean) / (self.y_std + self.epsilon)

		normalization_params = pd.DataFrame({
			"parameter": ["X_mean", "X_std", "y_mean", "y_std"],
			"value": [self.X_mean, self.X_std, self.y_mean, self.y_std]
		})
		normalization_params.to_csv(self.normalization_path, index=False)


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

	# Execute the training process and save the results.
	def run(self):
		self.train()
		self.save_model()
		print("Model:", self.model_path)
		print("theta0:", self.theta0)
		print("theta1:", self.theta1)
		print("")
		print("Normalization:\t", self.normalization_path)
		print("X_mean:\t\t", self.X_mean)
		print("X_std:\t\t", self.X_std)
		print("y_mean:\t\t", self.y_mean)
		print("y_std:\t\t", self.y_std)
		print("")
		print("Error ", self.recorded_errors[-1])
		print("")
		self.plot_training()
		print("Model trained and saved")
# LinregTrainer


# Parse command line arguments
def parse_args():
	parser = argparse.ArgumentParser(description='Run linear regression training.')
	parser.add_argument('--data', type=str, help='Path to the data CSV file', default="data/data.csv")
	parser.add_argument('--model', type=str, help='Path to save the model CSV file', default="data/model.csv")
	parser.add_argument('--norm', type=str, help='Path to save the normalization CSV file', default="data/normalization.csv")
	parser.add_argument('--epochs', type=int, help='Number of training epochs', default=500)
	parser.add_argument('--learning_rate', type=float, help='Gradient descent learning rate', default=0.01)
	parser.add_argument('--epsilon', type=float, help='Small value to prevent division by zero', default=1e-8)
	parser.add_argument('--plot', type=str, help='Path to save the training plot', default="plots/training_plot.png")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	trainer = LinregTrainer(data_path=args.data, model_path=args.model, normalization_path=args.norm, epochs=args.epochs, learning_rate=args.learning_rate, epsilon=args.epsilon, plot_path=args.plot)
	trainer.run()
