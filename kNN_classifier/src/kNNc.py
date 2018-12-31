import matplotlib.pyplot as plt
from random_forest_classifier.src.rfc_utils import *
from kNN_classifier.src.kNN_utils import *
from random_forest_classifier.src.rfc import plot_predictions
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-dark')
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1


def main(data_df):
    assert isinstance(data_df, pd.DataFrame)
    actuals = get_actuals(data_df)
    encoded_df = get_encoding(data_df)
    feat_vectors, features_names = get_features(encoded_df)

    scaler = StandardScaler()
    feat_vectors = pd.DataFrame(scaler.fit_transform(feat_vectors))

    train_features, test_features, train_actuals, test_actuals = get_split(
        feat_vectors, actuals)

    print("training model...")
    kNN_model = train_kNN_model(train_features, train_actuals)

    print("getting predictions...")
    test_pred = get_prediction(kNN_model, test_features)

    print("\nprinting summary:")
    print(classification_report(test_pred, test_actuals))
    plot_predictions(test_actuals, test_pred)


if __name__ == "__main__":
    file_name = "random_forest_classifier/files/breast_cancer.csv"
    data = pd.read_csv(file_name)
    main(data)
