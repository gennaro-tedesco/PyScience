import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from random_forest_classifier.src.rfc_utils import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_digits

plt.style.use('seaborn-dark')

def main(data_df):
    assert isinstance(data_df, pd.DataFrame)
    actuals = get_actuals(data_df)
    encoded_df = get_encoding(data_df)
    feat_vectors, features_names = get_features(encoded_df) 

    train_features, test_features, train_actuals, test_actuals = get_split(feat_vectors, actuals)

    print("training model...")
    rf_model = train_model(train_features, train_actuals)
    
    print("getting predictions...")
    test_pred = get_prediction(rf_model, train_features, test_features)

    
    print("\nprinting summary:")
    print_summary(rf_model, features_names, test_actuals, test_pred)
    plot_predictions(test_actuals, test_pred)


def print_summary(rf_model, features_names, test_actuals, test_pred):
    feature_importances = pd.Series(rf_model.feature_importances_,index=features_names).sort_values(ascending=False)
    print(classification_report(test_pred, test_actuals))
    print("\nfeatures importance")
    print(feature_importances)

def plot_predictions(test_actuals, test_pred, normalise=False):
    assert len(test_pred) == len(test_actuals)
    cm = confusion_matrix(test_actuals, test_pred)
    if normalise:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        
    sns.heatmap(cm, square=True, annot=True, cbar=True, cmap="Blues")
    plt.xlabel('actuals')
    plt.ylabel('predictions');
    plt.show()


if __name__ == "__main__":
    file_name = "random_forest_classifier/files/digits.csv"
    data = pd.read_csv(file_name)
    main(data)