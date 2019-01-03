import pandas as pd
import logging
logging.captureWarnings(True)
from src.utils import *
from src.bestclassifier import BestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

### -------------------------------------------------- ###
### change the below variables according to your needs ###
### -------------------------------------------------- ###

# the models that you want to compare
models = {
	'RandomForestClassifier': RandomForestClassifier(),
	'KNeighboursClassifier': KNeighborsClassifier(),
	'LogisticRegression': LogisticRegression()
}

# the optimisation parameters for each of the above models
params = {
	'RandomForestClassifier':{ 
			"n_estimators"      : [100, 200, 500, 1000],
			"max_features"      : ["auto", "sqrt", "log2"],
			"bootstrap": [True],
            "criterion": ['gini', 'entropy'],
            "oob_score": [True, False]
			},
	'KNeighboursClassifier': {
		'n_neighbors': np.arange(3, 15),
		'weights': ['uniform', 'distance'],
		'algorithm': ['ball_tree', 'kd_tree', 'brute']
		},
	'LogisticRegression': {
		'solver': ['newton-cg', 'sag', 'lbfgs'],
		'multi_class': ['ovr', 'multinomial']
		}  
}

# the data source
file_name = "files/iris.csv"

if __name__ == "__main__":
	data_df = pd.read_csv(file_name)
	actuals = get_classifier_actuals(data_df)
	encoded_df = get_classifier_encoding(data_df)
	feat_vectors, features_names = get_classifier_features(encoded_df) 

	train_features, test_features, train_actuals, test_actuals = get_split(feat_vectors, actuals)

	bc = BestClassifier(models, params, 'f1_weighted')
	bc.fit(train_actuals, train_features)
	bc.evaluation()
