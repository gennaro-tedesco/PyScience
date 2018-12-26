import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scikitplot as skplt
from random_forest_classifier.src.rfc_utils import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


plt.style.use('seaborn-dark')
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1


def main(data_df):
    assert isinstance(data_df, pd.DataFrame)
    actuals = get_actuals(data_df)
    encoded_df = get_encoding(data_df)
    feat_vectors, features_names = get_features(encoded_df) 

    train_features, test_features, train_actuals, test_actuals = get_split(feat_vectors, actuals)

    print("training model...")
    rf_model = train_model(train_features, train_actuals)
    
    print("getting predictions...")
    test_pred = get_prediction(rf_model, test_features)

    print("\nprinting summary:")
    print_summary(rf_model, features_names, test_actuals, test_pred)
    plot_predictions(test_actuals, test_pred)


def print_summary(rf_model, features_names, test_actuals, test_pred):
    feature_importances = pd.Series(rf_model.feature_importances_,index=features_names).sort_values(ascending=False)    
    print("\nfeatures importance")
    print(feature_importances)
    print(classification_report(test_pred, test_actuals))


def plot_predictions(test_actuals, test_pred):
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


if __name__ == "__main__":
    file_name = "random_forest_classifier/files/breast_cancer.csv"
    data = pd.read_csv(file_name)
    main(data)