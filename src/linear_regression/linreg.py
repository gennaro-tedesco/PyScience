import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from src.utils import * 

def linreg_main(data_df):
	assert isinstance(data_df, pd.DataFrame)
	actuals = get_regressor_actuals(data_df)
	encoded_df = get_regressor_encoding(data_df)
	feat_vectors, features_names = get_regressor_features(encoded_df) 

	train_features, test_features, train_actuals, test_actuals = get_split(feat_vectors, actuals)

	print("training linear regression...")
	estimator = LinearRegression()
	linreg_model = estimator.fit(train_features, train_actuals)
	test_pred = pd.Series(linreg_model.predict(test_features))

	test_r2_score = r2_score(test_actuals, test_pred)
	error = math.sqrt(mean_squared_error(test_actuals, test_pred))
	print("r2 score is: {}".format(test_r2_score))
	print("square root of residuals is: {}".format(error))
 
if __name__ == "__main__":    
	file_name = "files/housing.csv"
	data_df = pd.read_csv(file_name)
	linreg_main(data_df)