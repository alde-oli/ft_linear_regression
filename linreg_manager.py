import os
import sys
import argparse
from src.linreg_train import LinregTrainer
from src.linreg_predict import LinregPredictor
from src.data_model_plot import DataPlotter
from src.reset import input_reset_files


def	linreg_manager():
	if not os.path.exists("data"):
		os.makedirs("data")
	if not os.path.exists("plots"):
		os.makedirs("plots")
	while True:
		print(f"""
 ____________________________________________________________________
| choose an action to perform:
|
|     train                  (train the model)
|
|     predict                (predict a price)
|
|     plot                   (plot data and model)
|
|     reset                  (reset model and normalization files)
|
|     exit                   (exit the program)
|
 ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
""")
		action = input("action: ")
		if action == "train":
			elem =LinregTrainer.input_instanciate()
			if elem.abort:
				continue
			elem.run()
		elif action == "predict":
			elem = LinregPredictor.input_instanciate()
			if elem.abort:
				continue
			elem.run()
		elif action == "plot":
			elem = DataPlotter.input_instanciate()
			if elem.abort:
				continue
			elem.run()
		elif action == "reset":
			input_reset_files()
		elif action == "exit":
			sys.exit(0)
		else:
			print("Invalid action")


if __name__ == "__main__":
	linreg_manager()