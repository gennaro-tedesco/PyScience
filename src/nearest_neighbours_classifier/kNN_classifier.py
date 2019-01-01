import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from src.utils import * 

def train_kNN_model(estimator, train_features, train_actuals):
	assert isinstance(train_features, pd.DataFrame)
	assert isinstance(estimator, KNeighborsClassifier)	
	parameter_grid = {
		'n_neighbors': np.arange(3, 15),
		'weights': ['uniform', 'distance'],
		'algorithm': ['ball_tree', 'kd_tree', 'brute']
		}  

	knn_gscv = GridSearchCV(estimator, parameter_grid, cv=5)
	knn_gscv.fit(train_features, train_actuals)
	
	print("best parameters are: {}".format(knn_gscv.best_estimator_))
	print("best accuracy score is: {}".format(knn_gscv.best_score_))
	return knn_gscv.best_estimator_

def kNN_classifier_main(data_df):
	assert isinstance(data_df, pd.DataFrame)
	actuals = get_classifier_actuals(data_df)
	encoded_df = get_classifier_encoding(data_df)
	feat_vectors, features_names = get_classifier_features(encoded_df) 

	train_features, test_features, train_actuals, test_actuals = get_split(feat_vectors, actuals)

	print("training kNN classifier...")
	estimator = KNeighborsClassifier()
	kNN_model = train_kNN_model(estimator, train_features, train_actuals)
	test_pred = pd.Series(kNN_model.predict(test_features))

	print(classification_report(test_pred, test_actuals))
	plot_classifier_predictions(test_actuals, test_pred)