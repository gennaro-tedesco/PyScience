import pandas as pd
import forestci as fci
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from sklearn.model_selection import GridSearchCV

def get_encoding(df):
	assert isinstance(df, pd.DataFrame)
	features = df.drop('actuals', axis=1)
	return pd.get_dummies(features)


def get_actuals(df):
	assert isinstance(df, pd.DataFrame)
	return pd.Series(pd.factorize(df['actuals'])[0])


def get_features(df):
	assert isinstance(df, pd.DataFrame)
	features_names = list(df)
	return df, features_names


def get_split(features, actuals):
	assert isinstance(features, pd.DataFrame)
	assert isinstance(actuals, pd.Series)
	return train_test_split(features, actuals, test_size=0.25, random_state=0)


def train_model(train_features, train_actuals):
	assert isinstance(train_features, pd.DataFrame)

	estimator = RandomForestClassifier()
	param_grid = { 
			"n_estimators"      : [100, 200, 500, 1000],
			"max_features"      : ["auto", "sqrt", "log2"],
			"bootstrap": [True, False]
			}

	grid_model = GridSearchCV(estimator, param_grid, n_jobs=-1, cv=5)
	grid_model.fit(train_features, train_actuals)
	print("best parameters are: {}".format(grid_model.best_score_))
	print("best accuracy score is: {}".format(grid_model.best_estimator_))
	return grid_model.best_estimator_


def get_prediction(model, test_features):
	return pd.Series(model.predict(test_features))
