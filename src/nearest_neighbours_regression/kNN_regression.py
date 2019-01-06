import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from src.utils import * 

def train_kNN_model(estimator, train_features, train_actuals):
	assert isinstance(train_features, pd.DataFrame)
	assert isinstance(estimator, KNeighborsRegressor)	
	parameter_grid = {
		'n_neighbors': np.arange(3, 15),
		'weights': ['uniform', 'distance'],
		'algorithm': ['ball_tree', 'kd_tree', 'brute']
		}  

	knn_gscv = GridSearchCV(estimator, parameter_grid, cv=10, scoring='r2')
	knn_gscv.fit(train_features, train_actuals)
	
	print("best parameters are: {}".format(knn_gscv.best_estimator_))
	print("best validation r2 score is: {}".format(knn_gscv.best_score_))
	return knn_gscv.best_estimator_

def kNN_regression_main(data_df):
	assert isinstance(data_df, pd.DataFrame)
	actuals = get_regressor_actuals(data_df)
	encoded_df = get_regressor_encoding(data_df)
	feat_vectors, features_names = get_regressor_features(encoded_df) 

	train_features, test_features, train_actuals, test_actuals = get_split(feat_vectors, actuals)
	train_features, test_features = get_scaling(train_features, test_features)
	
	print("training kNN classifier...")
	estimator = KNeighborsRegressor()
	kNN_model = train_kNN_model(estimator, train_features, train_actuals)
	test_pred = pd.Series(kNN_model.predict(test_features))

	print_regression_summary(test_actuals, test_pred)
	plot_regressor_predictions(test_actuals, test_pred)
