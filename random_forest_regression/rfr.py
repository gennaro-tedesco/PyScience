import math
import matplotlib.pyplot as plt
import numpy as np
from random_forest_regression.rfr_utils import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

plt.style.use('seaborn')

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
    test_pred, error_pred = get_prediction(rf_model, train_features, test_features)

    print("\nprinting summary:")
    print_summary(rf_model, features_names, test_actuals, test_pred)
    #plot_predictions(test_actuals, test_pred, error_pred)


def print_summary(rf_model, features_names, test_actuals, test_pred):
    test_r2_score = r2_score(test_actuals, test_pred)
    error = math.sqrt(mean_squared_error(test_actuals, test_pred))
    feature_importances = pd.Series(rf_model.feature_importances_,index=features_names).sort_values(ascending=False)
    print("r2 score is: {}".format(test_r2_score))
    print("square root of residuals is: {}".format(error))
    print("\nfeatures importance:")
    print(feature_importances)

def plot_predictions(test_actuals, test_pred, errors):
    assert len(test_pred) == len(test_actuals)
    plt.errorbar(list(range(0, len(test_actuals))), test_actuals, yerr=np.sqrt(errors), fmt='bo-', label='actual')
    plt.plot(list(range(0, len(test_pred))), test_pred, 'ro-', label='pred')
    #plt.errorbar(test_actuals, test_pred, yerr=np.sqrt(errors), fmt='o')
    #plt.plot([min(test_actuals), max(test_actuals)], [min(test_actuals), max(test_actuals)], 'k--')
    plt.show()


if __name__ == "__main__":
    file_name = "random_forest_regression/temperatures.csv"
    data = pd.read_csv(file_name)
    main(data)
