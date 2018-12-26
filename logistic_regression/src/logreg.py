import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from random_forest_classifier.src.rfc_utils import *
from random_forest_classifier.src.rfc import plot_predictions
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler


def main(data_df):
    actuals = get_actuals(data_df)
    encoded_df = get_encoding(data_df)
    feat_vectors, features_names = get_features(encoded_df) 

    scaler = StandardScaler()
    feat_vectors = pd.DataFrame(scaler.fit_transform(feat_vectors))

    train_features, test_features, train_actuals, test_actuals = get_split(feat_vectors, actuals)

    print("training model...")
    logreg_model = LogisticRegression(solver='lbfgs',
                                    multi_class='multinomial')
    logreg_model.fit(train_features, train_actuals)

    print("getting predictions...")
    test_pred = logreg_model.predict(test_features)

    print("printing summary")
    print(classification_report(test_pred, test_actuals))
    plot_predictions(test_actuals, test_pred)

if __name__ == "__main__":    
    file_name = "random_forest_classifier/files/breast_cancer.csv"
    data_df = pd.read_csv(file_name)
    main(data_df)