import pandas as pd
import forestci as fci
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from itertools import product

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


def train_model(train_vectors, train_actuals):
	assert isinstance(train_vectors, pd.DataFrame)
	parameter_grid = [
		(100, 200, 500, 1000),
		(0.3, 0.4)
	]
	best_score = float("-inf")

	for n, f in product(*parameter_grid):
		print("training on n_estimators={} and max_features={}".format(n, f))
		est = RandomForestClassifier(oob_score=True,
									n_estimators=n,
									max_features=f)
		est.fit(train_vectors, train_actuals)
		if est.oob_score_ > best_score:
			best_n_estimators = n
			best_max_features = f
			best_score, best_est = est.oob_score_, est
	print("best n_estimators={}, best_max_features={}"
		.format(best_n_estimators, best_max_features))
	return est


def get_prediction(model, test_features):
	return pd.Series(model.predict(test_features))
