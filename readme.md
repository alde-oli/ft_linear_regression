to install requirements: 
	python install -r requirements.txt


project split in 3 programs:
	-linreg_train.py:
		trains model on a dataset using linear regression and gradient descent
		arguments:
			--data:  dataset used to train the model (default: data/data.csv)
			--model: model file (default: model.csv), parameters of the model are theta0 and theta1, with formula y = theta0 + theta1*x
			--norm: normalization file (default: norm.csv), parameters of the normalization are mean and std, with formula x = (x - mean) / std
			--epochs: number of epochs (cycles) to train the model (default: 500)
			--learning_rate: learning rate of the gradient descent (default: 0.01), defines how much the model is updated at each step
			--epsilon: used to avoid division by zero in normalization (default: 1e-10) when std is close to zero
			--plot: plot the evolution of the error during training
		python linreg_train [--data DATA] [--model MODEL] [--norm NORM] [--epochs EPOCHS] [--learning_rate LEARNING_RATE] [--epsilon EPSILON] [--plot PLOT]

	-linreg_predict:
		predicts values using a trained model using user input
		arguments:
			--model: model file (default: model.csv), parameters of the model are theta0 and theta1, with formula y = theta0 + theta1*x
			--norm: normalization file (default: norm.csv), parameters of the normalization are mean and std, with formula x = (x - mean) / std
			--epsilon: used to avoid division by zero in normalization (default: 1e-10) when std is close to zero
		python linreg_predict [--data DATA] [--model MODEL] [--norm NORM] [--epsilon EPSILON]
	
	-data_model_plot.py:
		plots the dataset with the model function
		arguments:
			--data: dataset used to train the model (default: data/data.csv)
			--model: model file (default: model.csv), parameters of the model are theta0 and theta1, with formula y = theta0 + theta1*x
			--norm: normalization file (default: norm.csv), parameters of the normalization are mean and std, with formula x = (x - mean) / std
			--plot: plot the dataset and the model function
			--epsilon: used to avoid division by zero in normalization (default: 1e-10) when std is close to zero
		python data_model_plot [--data DATA] [--model MODEL] [--norm NORM] [--plot PLOT] [--epsilon EPSILON]


clear model and plots:
	python reset.py [--model MODEL] [--norm NORM] [--data_plot DATA_PLOT] [--training_plot TRAINING_PLOT]