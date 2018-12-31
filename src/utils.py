import matplotlib.pyplot as plt
import pandas as pd 
import scikitplot as skplt
import seaborn as sns
import numpy as np 
import math 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from src.cyclic_encoder.ce import * 

plt.style.use('seaborn-dark')
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["lines.markersize"] = 3

def get_regressor_encoding(df):
	assert isinstance(df, pd.DataFrame)
	return pd.get_dummies(df)

def get_regressor_actuals(df):
	assert isinstance(df, pd.DataFrame)
	return df['actuals']

def get_classifier_encoding(df):
	assert isinstance(df, pd.DataFrame)
	features = df.drop('actuals', axis=1)
	return pd.get_dummies(features)

def get_classifier_actuals(df):
	assert isinstance(df, pd.DataFrame)
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

## ---------------
## print summaries
## ---------------
def print_classifier_summary(rf_model, features_names, test_actuals, test_pred):
    feature_importances = pd.Series(rf_model.feature_importances_, index=features_names).sort_values(ascending=False)    
    print("\nfeatures importance")
    print(feature_importances)
    print(classification_report(test_pred, test_actuals))

def print_rfr_summary(rf_model, features_names, test_actuals, test_pred):
    test_r2_score = r2_score(test_actuals, test_pred)
    error = math.sqrt(mean_squared_error(test_actuals, test_pred))
    feature_importances = pd.Series(rf_model.feature_importances_, index=features_names).sort_values(ascending=False)
    print("r2 score is: {}".format(test_r2_score))
    print("square root of residuals is: {}".format(error))
    print("\nfeatures importance")
    print(feature_importances)

## -----
## plots 
## -----
def plot_classifier_predictions(test_actuals, test_pred):
    assert len(test_pred) == len(test_actuals)
    fig, (ax1, ax2) = plt.subplots(1,2)   
    sns.heatmap(precision_recall_fscore_support(test_actuals, test_pred)[:-1],
		annot=True, 
		cbar=False, 
		xticklabels=list(np.unique(test_actuals)), 
		yticklabels=['recall', 'precision', 'f1-score'],
		ax=ax1,
		cmap='Blues')
    ax1.set_title('classification report') 

    skplt.metrics.plot_confusion_matrix(test_actuals, test_pred, 
					normalize=True, 
					ax=ax2)

    plt.tight_layout()
    plt.show()


def plot_regressor_predictions(test_actuals, test_pred, errors):
    assert len(test_pred) == len(test_actuals)
    plt.subplot(2, 1, 1)
    plt.errorbar(list(range(0, len(test_actuals))), test_actuals, 
		yerr=np.sqrt(errors), 
		fmt='bo-', 
		label='actual')
    plt.plot(list(range(0, len(test_pred))), test_pred, 
	    'ro-', 
	    label='pred')
    plt.title('predictions and actuals')
    plt.xlabel('index of observations')
    plt.ylabel('predictions value')   
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.errorbar(test_actuals, test_pred, 
		yerr=np.sqrt(errors), 
		fmt='o')
    plt.plot([min(test_actuals), max(test_actuals)], [min(test_actuals), max(test_actuals)], 'k--')
    plt.xlabel('actuals')
    plt.ylabel('predictions')

    plt.tight_layout()
    plt.show()