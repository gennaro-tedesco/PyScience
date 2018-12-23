import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def get_encoding(df):
	assert isinstance(df, pd.DataFrame)
	return pd.get_dummies(df)


def get_actuals(df):
	assert isinstance(df, pd.DataFrame)
	return df['actual']


def get_features(df):
	assert isinstance(df, pd.DataFrame)
	features = df.drop('actual', axis=1)
	features_names = list(features)
	return features, features_names


def get_split(features, actuals):
	assert isinstance(features, pd.DataFrame)
	assert isinstance(actuals, pd.Series)
	return train_test_split(features, actuals, test_size=0.25)


def train_model(train_vectors, train_labels):
	assert isinstance(train_vectors, pd.DataFrame)
	rf = RandomForestRegressor(n_estimators=1000)
	rf.fit(train_vectors, train_labels)
	return rf


def get_prediction(model, test_vectors):
	return pd.Series(model.predict(test_vectors))
