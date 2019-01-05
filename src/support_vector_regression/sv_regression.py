import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from src.utils import * 

def train_SVR_model(estimator, train_features, train_actuals):
	assert isinstance(train_features, pd.DataFrame)
	assert isinstance(estimator, SVR)	
	parameter_grid = {
		'kernel':['rbf', 'linear', 'sigmoid'],
		'gamma': ['auto', 'scale']
		}  

	svr_gscv = GridSearchCV(estimator, parameter_grid, cv=10, scoring='r2')
	svr_gscv.fit(train_features, train_actuals)
	
	print("best parameters are: {}".format(svr_gscv.best_estimator_))
	print("best r2 score is: {}".format(svr_gscv.best_score_))
	return svr_gscv.best_estimator_

def sv_regression_main(data_df):
	assert isinstance(data_df, pd.DataFrame)
	actuals = get_regressor_actuals(data_df)
	encoded_df = get_regressor_encoding(data_df)
	feat_vectors, features_names = get_classifier_features(encoded_df) 

	train_features, test_features, train_actuals, test_actuals = get_split(feat_vectors, actuals)

	print("training SV regression...")
	estimator = SVR()
	svr_model = train_SVR_model(estimator, train_features, train_actuals)
	test_pred = pd.Series(svr_model.predict(test_features))

	test_r2_score = r2_score(test_actuals, test_pred)
	error = math.sqrt(mean_squared_error(test_actuals, test_pred))
	print("r2 score is: {}".format(test_r2_score))
	print("square root of residuals is: {}".format(error))