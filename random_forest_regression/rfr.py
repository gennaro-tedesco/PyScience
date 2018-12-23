import math
from random_forest_regression.rfr_utils import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def main(data_df):
    assert isinstance(data_df, pd.DataFrame)
    encoded_df = get_encoding(data)

    actuals = get_actuals(encoded_df)
    feat_vectors, features_names = get_features(encoded_df)
    train_features, test_features, train_actuals, test_actuals = get_split(
        feat_vectors, actuals)

    print("training model...")
    rf_model = train_model(train_features, train_actuals)

    print("getting predictions...")
    test_pred = get_prediction(rf_model, test_features)

    print("printing summary:")
    print("----------------")
    test_r2_score = r2_score(test_actuals, test_pred)
    error = math.sqrt(mean_squared_error(test_actuals, test_pred))
    importances = list(rf_model.feature_importances_)
    feature_importances = [(feature, round(importance, 3))
                           for feature, importance in zip(features_names, importances)]
    feature_importances = sorted(
        feature_importances, key=lambda x: x[1], reverse=True)
    print("r2 score is: {}".format(test_r2_score))
    print("square root of residuals is: {}".format(error))
    print("\nfeatures importance:")
    [print('{:20}: {}'.format(*pair)) for pair in feature_importances]

    plot_predictions(test_actuals, test_pred)


def plot_predictions(test_actuals, test_pred):
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    assert len(test_pred) == len(test_actuals)
    plt.plot(list(range(0, len(test_actuals))),
             test_actuals, 'b-', label='actual')
    plt.plot(list(range(0, len(test_pred))), test_pred, 'r-', label='pred')
    plt.show()


if __name__ == "__main__":
    file_name = "random_forest_regression/temperatures.csv"
    data = pd.read_csv(file_name)
    main(data)
