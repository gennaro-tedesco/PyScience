import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from src.utils import * 

def train_logreg_model(estimator, train_features, train_actuals):
	assert isinstance(train_features, pd.DataFrame)
	assert isinstance(estimator, LogisticRegression)	
	parameter_grid = {
		'solver': ['newton-cg', 'sag', 'lbfgs'],
		'multi_class': ['ovr', 'multinomial']
		}  

	logreg_gscv = GridSearchCV(estimator, parameter_grid, cv=5)
	logreg_gscv.fit(train_features, train_actuals)
	
	print("best parameters are: {}".format(logreg_gscv.best_estimator_))
	print("best accuracy score is: {}".format(logreg_gscv.best_score_))
	return logreg_gscv.best_estimator_

def logreg_main(data_df):
	assert isinstance(data_df, pd.DataFrame)
	actuals = get_classifier_actuals(data_df)
	encoded_df = get_classifier_encoding(data_df)
	feat_vectors, features_names = get_classifier_features(encoded_df) 

	train_features, test_features, train_actuals, test_actuals = get_split(feat_vectors, actuals)

	print("training kNN classifier...")
	estimator = LogisticRegression()
	kNN_model = train_logreg_model(estimator, train_features, train_actuals)
	test_pred = pd.Series(kNN_model.predict(test_features))

	print(classification_report(test_pred, test_actuals))
	plot_classifier_predictions(test_actuals, test_pred)


if __name__ == "__main__":    
    file_name = "files/breast_cancer.csv"
    data_df = pd.read_csv(file_name)
    logreg_main(data_df)