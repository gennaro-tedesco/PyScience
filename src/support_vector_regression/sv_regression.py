import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from src.utils import * 
from sklearn.preprocessing import Normalizer


def train_SVR_model(estimator, train_features, train_actuals):
	assert isinstance(train_features, pd.DataFrame)
	assert isinstance(estimator, SVR)	
	parameter_grid = {
		'kernel':['rbf', 'linear', 'sigmoid'],
		'gamma': ['auto', 'scale'],
		'epsilon': [0.05, 0.1, 0.15]
		}  

	svr_gscv = GridSearchCV(estimator, parameter_grid, cv=5, scoring='r2')
	svr_gscv.fit(train_features, train_actuals)
	
	print("best parameters are: {}".format(svr_gscv.best_estimator_))
	print("best validation r2 score is: {}".format(svr_gscv.best_score_))
	return svr_gscv.best_estimator_

def sv_regression_main(data_df):
	assert isinstance(data_df, pd.DataFrame)
	actuals = get_regressor_actuals(data_df)
	encoded_df = get_regressor_encoding(data_df)
	feat_vectors, features_names = get_regressor_features(encoded_df) 

	train_features, test_features, train_actuals, test_actuals = get_split(feat_vectors, actuals)
	train_features, test_features = get_scaling(train_features, test_features)
	scaler = StandardScaler()
	train_actuals = scaler.fit_transform(train_actuals.values.reshape(-1, 1))

	print("training SV regression...")
	estimator = SVR()
	svr_model = train_SVR_model(estimator, train_features, train_actuals)
	test_pred = pd.Series(scaler.inverse_transform(pd.Series(svr_model.predict(test_features))))

	print_regression_summary(test_actuals, test_pred)
	plot_regressor_predictions(test_actuals, test_pred)