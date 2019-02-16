import pandas as pd
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from src.utils import *

def base_model(input_dim, first_layer, second_layer):
	assert isinstance(input_dim, int)
	assert isinstance(first_layer, int)
	assert isinstance(second_layer, int)
	model = Sequential()
	model.add(Dense(first_layer, input_dim=input_dim, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(second_layer, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer = 'adam', metrics=['mean_squared_error']) 
	return model

def train_kerasNN_model(estimator, train_features, train_actuals):
	assert isinstance(train_features, pd.DataFrame)
	assert isinstance(estimator, KerasRegressor)	
	parameter_grid = {
		'first_layer': [10, 50, 100],
		'second_layer': [5, 10, 20],
		'epochs': [100],
		'batch_size': [10, 50]
		}  
		
	keras_gscv = GridSearchCV(estimator, parameter_grid, cv=5, scoring='r2', n_jobs=1, verbose=5)
	keras_gscv.fit(train_features, train_actuals) 

	print("best validation mean_squared_error is: {}".format(keras_gscv.best_score_))
	means = pd.Series(keras_gscv.cv_results_['mean_test_score'])
	stds = pd.Series(keras_gscv.cv_results_['std_test_score'])
	params = pd.Series(keras_gscv.cv_results_['params'])
	scores_df = pd.concat([means, stds, params], axis=1)
	scores_df.columns = ['score_mean', 'score_std', 'params']
	scores_df = scores_df.sort_values(by=['score_mean'], ascending=False).reset_index()
	print("best parameters are: {}".format(scores_df['params'][0]))
	return keras_gscv.best_estimator_

def plot_loss(history):
	epochs = range(1, len(history.history['loss']) + 1)
	plt.plot(epochs, history.history['loss'], 'b')
	plt.title('Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.savefig('gallery/NN_loss.png')

def kerasNN_regression_main(data_df):
	assert isinstance(data_df, pd.DataFrame)
	actuals = get_regressor_actuals(data_df)
	encoded_df = get_regressor_encoding(data_df)
	feat_vectors, features_names = get_regressor_features(encoded_df) 

	train_features, test_features, train_actuals, test_actuals = get_split(feat_vectors, actuals)
	train_features, test_features = get_scaling(train_features, test_features)
	scaler = StandardScaler()
	train_actuals = scaler.fit_transform(train_actuals.values.reshape(-1, 1))
	train_actuals = pd.Series(train_actuals.flatten())

	print("training neural network regression...")
	estimator = KerasRegressor(build_fn=base_model, input_dim=len(list(train_features)))
	kerasNN_model = train_kerasNN_model(estimator, train_features, train_actuals)
	history = kerasNN_model.model.model.history
	plot_loss(history)
	
	train_features, test_features = get_scaling(train_features, test_features)
	test_pred = pd.Series(scaler.inverse_transform(pd.Series(kerasNN_model.predict(test_features))))

	print_regression_summary(test_actuals, test_pred)
	plot_regressor_predictions(test_actuals, test_pred)
