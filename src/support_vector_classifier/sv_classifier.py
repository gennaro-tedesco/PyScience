import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from src.utils import * 

def train_SVC_model(estimator, train_features, train_actuals):
	assert isinstance(train_features, pd.DataFrame)
	assert isinstance(estimator, SVC)	
	parameter_grid = {
		'kernel':['rbf', 'linear', 'poly', 'sigmoid'],
		'gamma': ['auto', 'scale']
		}  

	svc_gscv = GridSearchCV(estimator, parameter_grid, cv=5, scoring='f1_weighted')
	svc_gscv.fit(train_features, train_actuals)
	
	print("best parameters are: {}".format(svc_gscv.best_estimator_))
	print("best f1_weighted score is: {}".format(svc_gscv.best_score_))
	return svc_gscv.best_estimator_

def sv_classifier_main(data_df):
	assert isinstance(data_df, pd.DataFrame)
	actuals = get_classifier_actuals(data_df)
	encoded_df = get_classifier_encoding(data_df)
	feat_vectors, features_names = get_classifier_features(encoded_df) 

	train_features, test_features, train_actuals, test_actuals = get_split(feat_vectors, actuals)
	train_features, test_features = get_scaling(train_features, test_features)

	print("training SV classifier...")
	estimator = SVC()
	svc_model = train_SVC_model(estimator, train_features, train_actuals)
	test_pred = pd.Series(svc_model.predict(test_features))

	print(classification_report(test_pred, test_actuals))
	plot_classifier_predictions(test_actuals, test_pred)