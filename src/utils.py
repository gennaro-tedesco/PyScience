import matplotlib.pyplot as plt
import pandas as pd 
import scikitplot as skplt
import seaborn as sns
import numpy as np 
import math 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from src.cyclic_encoder.ce import get_cyclicl_encoding
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

plt.style.use('seaborn-dark')
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["lines.markersize"] = 3


def get_encoder_inst(feature_col):
    """
    returns: an instance of sklearn OneHotEncoder fit against a (training) column feature
    such instance is saved and can then be loaded to transform unseen data
    """
    assert isinstance(feature_col, pd.Series)
    feature_vec = feature_col.sort_values().values.reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(feature_vec) 
    with open('encoders/' + feature_col.name + '_enc_inst.pkl', 'wb') as output_file:
            pickle.dump(enc, output_file)
    return enc 

def get_one_hot_enc(feature_col, enc):
    """
    maps an unseen column feature using one-hot-encoding previously fit against training data 
    returns: a pd.DataFrame of newly one-hot-encoded feature
    """
    assert isinstance(feature_col, pd.Series)
    assert isinstance(enc, OneHotEncoder)
    unseen_vec = feature_col.values.reshape(-1, 1)
    encoded_vec = enc.transform(unseen_vec).toarray()
    encoded_df = pd.DataFrame(encoded_vec)
    cat_levels = [item for sublist in enc.categories_ for item in sublist]
    encoded_df.columns = [feature_col.name + '_' + level for level in cat_levels]
    return encoded_df

def encode_train_df(df):
    """
    returns a data frame where categorical variables (or better, variables of 
    type object) are encoded
    """
    assert isinstance(df, pd.DataFrame)
    dt = df.copy().reset_index()
    columns_to_encode = df.select_dtypes(include=['object']).columns
    for feature in columns_to_encode:
        print("encoding feature: {}".format(feature))
        feature_enc_inst = get_encoder_inst(dt[feature])
        encoded_feature_df = get_one_hot_enc(dt[feature], feature_enc_inst)
        dt = pd.concat([dt, encoded_feature_df], axis=1)
        dt.drop(feature, axis=1, inplace=True)
    dt = dt.reindex(sorted(dt.columns), axis=1)
    dt.drop('index', axis=1, inplace=True)
    return dt

def get_regressor_encoding(df):
	assert isinstance(df, pd.DataFrame)
	return encode_train_df(df)

def get_regressor_actuals(df):
	assert isinstance(df, pd.DataFrame)
	assert 'actuals' in list(df), "actuals column not present in the data set"
	return df['actuals']

def get_classifier_encoding(df):
	assert isinstance(df, pd.DataFrame)
	features = df.drop('actuals', axis=1)
	return encode_train_df(features)

def get_classifier_actuals(df):
	assert isinstance(df, pd.DataFrame)
	assert 'actuals' in list(df), "actuals column not present in the data set"
	return df['actuals']

def get_regressor_features(df):
	assert isinstance(df, pd.DataFrame)
	features = df.drop('actuals', axis=1)
	#features = get_cyclicl_encoding(features, 'day', 31) # drop if no date time objects
	#features = get_cyclicl_encoding(features, 'month', 12) # drop if no date time objects
	#features = features.drop('month', axis=1)
	#features = features.drop('day', axis=1)
	features_names = list(features)
	return features, features_names

def get_classifier_features(df):
	assert isinstance(df, pd.DataFrame)
	features_names = list(df)
	return df, features_names

def get_split(features, actuals):
	assert isinstance(features, pd.DataFrame)
	assert isinstance(actuals, pd.Series)
	return train_test_split(features, actuals, test_size=0.25)

def get_scaling(train_features, test_features):
	print("scaling numerical features...")
	assert isinstance(train_features, pd.DataFrame)
	assert isinstance(test_features, pd.DataFrame)
	columns_to_scale = train_features.select_dtypes(include=['float64', 'int']).columns

	scaler = StandardScaler()
	scaler.fit(train_features[columns_to_scale])
	train_features[columns_to_scale] = scaler.transform(train_features[columns_to_scale])
	test_features[columns_to_scale] = scaler.transform(test_features[columns_to_scale])
	return train_features, test_features


## ---------------
## print summaries
## ---------------
def print_classifier_summary(classifier_model, features_names, test_actuals, test_pred):
	feature_importances = pd.Series(classifier_model.feature_importances_, index=features_names).sort_values(ascending=False)    
	print("\nfeatures importance")
	print(feature_importances)
	print(classification_report(test_pred, test_actuals))

def print_regression_summary(test_actuals, test_pred):
	test_r2_score = r2_score(test_actuals, test_pred)
	error = math.sqrt(mean_squared_error(test_actuals, test_pred))	
	print("\nr2 score on test set is: {}".format(test_r2_score))
	print("square root of residuals is: {}".format(error))

## -----
## plots 
## -----
def plot_scaling(train_features, scaled_train_features):
	fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
	columns_to_scale = train_features.select_dtypes(include=['float64', 'int'])

	ax1.set_title('Before Scaling')
	for num_variable in columns_to_scale:
		sns.kdeplot(train_features[num_variable], ax=ax1)

	ax2.set_title('After Standard Scaler')
	for num_variable in columns_to_scale:
		sns.kdeplot(scaled_train_features[num_variable], ax=ax2)
	plt.show()

def plot_classifier_predictions(test_actuals, test_pred):
	assert len(test_pred) == len(test_actuals)
	fig, (ax1, ax2) = plt.subplots(1,2)   
	sns.heatmap(precision_recall_fscore_support(test_actuals, test_pred)[:-1],
		annot=True, 
		cbar=False, 
		xticklabels=list(np.unique(test_actuals)), 
		yticklabels=['recall', 'precision', 'f1-score'],
		ax=ax1,
		cmap='Blues',
		linewidths=0.5,
		square=True)
	ax1.set_title('classification report') 

	skplt.metrics.plot_confusion_matrix(test_actuals, test_pred, 
					normalize=True, 
					x_tick_rotation=45,
					ax=ax2)

	plt.tight_layout()
	plt.show()

def plot_regressor_predictions(test_actuals, test_pred):
	assert len(test_pred) == len(test_actuals)
	plt.subplot(2, 1, 1)
	plt.plot(list(range(0, len(test_actuals))), test_actuals, 
		'bo-', 
		label='actual')
	plt.plot(list(range(0, len(test_pred))), test_pred, 
		'ro-', 
		label='pred')
	plt.title('predictions and actuals')
	plt.xlabel('index of observations')
	plt.ylabel('predictions value')   
	plt.legend(loc='best')

	plt.subplot(2, 1, 2)
	plt.plot(test_actuals, test_pred, 'bo')
	plt.plot([min(test_actuals), max(test_actuals)], [min(test_actuals), max(test_actuals)], 'k--')
	plt.xlabel('actuals')
	plt.ylabel('predictions')

	plt.tight_layout()
	plt.show()