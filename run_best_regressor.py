import pandas as pd
import logging
logging.captureWarnings(True)
from src.utils import *
from src.bestregressor import BestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


### -------------------------------------------------- ###
### change the below variables according to your needs ###
### -------------------------------------------------- ###

# the models that you want to compare
models = {
	'RandomForestRegressor': RandomForestRegressor(),
	'LinearRegression': LinearRegression(),
	'KNeighboursRegressor': KNeighborsRegressor(),
	'SupportVectorRegressor': SVR()
}

# the optimisation parameters for each of the above models
params = {
	'RandomForestRegressor':{ 
			"n_estimators"      : [100, 200, 500, 1000],
			"max_features"      : ["auto", "sqrt", "log2"],
			"bootstrap": [True],
            "oob_score": [True, False]
			},
	'KNeighboursRegressor': {
		'n_neighbors': np.arange(3, 15),
		'weights': ['uniform', 'distance'],
		'algorithm': ['ball_tree', 'kd_tree', 'brute']
		},
	'LinearRegression': {
		'fit_intercept': [True, False],
		'normalize':  [True, False]
		},
	'SupportVectorRegressor': {
		'kernel':['rbf', 'linear', 'sigmoid'],
		'gamma': ['auto', 'scale'],
		'epsilon': [0.05, 0.1, 0.15]
		}  
}

# the data source
file_name = "datasets/housing.csv"

if __name__ == "__main__":
	data_df = pd.read_csv(file_name)
	actuals = get_regressor_actuals(data_df)
	encoded_df = get_regressor_encoding(data_df)
	feat_vectors, features_names = get_regressor_features(encoded_df) 

	train_features, test_features, train_actuals, test_actuals = get_split(feat_vectors, actuals)

	bc = BestRegressor(models, params, 'r2')
	bc.fit(train_actuals, train_features)
	bc.evaluation()
