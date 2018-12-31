import pandas as pd
import forestci as fci
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from itertools import product
from cyclic_encoder.ce import *
from sklearn.model_selection import GridSearchCV

def get_encoding(df):
	assert isinstance(df, pd.DataFrame)
	return pd.get_dummies(df)


def get_actuals(df):
	assert isinstance(df, pd.DataFrame)
	return df['actual']


def get_features(df):
	assert isinstance(df, pd.DataFrame)
	features = df.drop('actual', axis=1)
	features = get_cyclicl_encoding(features, 'day', 31)
	features = get_cyclicl_encoding(features, 'month', 12)
	features = features.drop('month', axis=1)
	features = features.drop('day', axis=1)
	features_names = list(features)
	return features, features_names


def get_split(features, actuals):
	assert isinstance(features, pd.DataFrame)
	assert isinstance(actuals, pd.Series)
	return train_test_split(features, actuals, test_size=0.25)


def train_model(train_features, train_actuals):
	assert isinstance(train_features, pd.DataFrame)
	
	estimator = RandomForestRegressor()
	param_grid = { 
			"n_estimators"      : [100, 200, 500, 1000],
			"max_features"      : ["auto", "sqrt", "log2"]
			}

	grid_model = GridSearchCV(estimator, param_grid, n_jobs=-1, cv=5)
	grid_model.fit(train_features, train_actuals)
	print("best parameters are: {}".format(grid_model.best_score_))
	print("best accuracy score is: {}".format(grid_model.best_estimator_))
	return grid_model.best_estimator_


def get_prediction(model, train_features, test_features):
	error_pred = fci.random_forest_error(model, train_features, test_features)
	return pd.Series(model.predict(test_features)), error_pred
