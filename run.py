import shutil
import pandas as pd
import logging
logging.captureWarnings(True)
from src.linear_regression.linreg import linreg_main
from src.logistic_regression.logreg import logreg_main 
from src.nearest_neighbours_classifier.kNN_classifier import kNN_classifier_main
from src.nearest_neighbours_regression.kNN_regression import kNN_regression_main 
from src.random_forest_classifier.rf_classifier import rfc_main
from src.random_forest_regression.rf_regression import rfr_main
from src.support_vector_classifier.sv_classifier import sv_classifier_main
from src.support_vector_regression.sv_regression import sv_regression_main


columns = shutil.get_terminal_size().columns

fun_map = {
    "linreg": linreg_main,
    "logreg": logreg_main,
    "kNN_classifier": kNN_classifier_main,
    "kNN_regression": kNN_regression_main,
    "random_forest_classifier": rfc_main,
    "random_forest_regression": rfr_main,
    "support_vector_classifier": sv_classifier_main,
    "support_vector_regression": sv_regression_main
}

### -------------------------------------------------- ###
### change the below variables according to your needs ###
### -------------------------------------------------- ###
file_name = "datasets/regression/mpgcars.csv" 
method = "random_forest_regression"

if __name__ == "__main__":
    print('-'*shutil.get_terminal_size().columns)
    print("executing {} on file '{}'".format(method, file_name).center(columns))
    print('-'*shutil.get_terminal_size().columns)

    fun_map[method](pd.read_csv(file_name))