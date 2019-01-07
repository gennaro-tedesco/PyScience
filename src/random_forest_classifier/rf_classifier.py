import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from src.utils import *


def train_rfc_model(estimator, train_features, train_actuals):
	assert isinstance(train_features, pd.DataFrame)
	assert isinstance(estimator, RandomForestClassifier)	
	param_grid = { 
			"n_estimators"      : [100, 200, 500, 1000],
			"max_features"      : ["auto", "sqrt", "log2"],
			"bootstrap": [True],
            "criterion": ['gini', 'entropy'],
            "oob_score": [True, False]
			}

	rfc_gscv = GridSearchCV(estimator, param_grid, n_jobs=-1, cv=10, scoring='f1_weighted')
	rfc_gscv.fit(train_features, train_actuals)
	print("best parameters are: {}".format(rfc_gscv.best_estimator_))
	print("best accuracy score is: {}".format(rfc_gscv.best_score_))
	return rfc_gscv.best_estimator_

def rfc_main(data_df):
    assert isinstance(data_df, pd.DataFrame)
    actuals = get_classifier_actuals(data_df)
    encoded_df = get_classifier_encoding(data_df)
    feat_vectors, features_names = get_classifier_features(encoded_df) 

    train_features, test_features, train_actuals, test_actuals = get_split(feat_vectors, actuals)
    train_features, test_features = get_scaling(train_features, test_features)

    print("training random forest classifier...")
    estimator = RandomForestClassifier()
    rfc_model = train_rfc_model(estimator, train_features, train_actuals)    
    test_pred = pd.Series(rfc_model.predict(test_features))

    print_classifier_summary(rfc_model, features_names, test_actuals, test_pred)
    plot_classifier_predictions(test_actuals, test_pred)
