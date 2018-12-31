import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def train_kNN_model(train_features, train_actuals):
    assert isinstance(train_features, pd.DataFrame)

    parameter_grid = {'n_neighbors': np.arange(3, 15)}  

    knn_model = KNeighborsClassifier()
    knn_gscv = GridSearchCV(knn_model, parameter_grid, cv=5)
    knn_gscv.fit(train_features, train_actuals)
    
    print("best parameters are: {}".format(knn_gscv.best_estimator_))
    print("best accuracy score is: {}".format(knn_gscv.best_score_))
    return knn_gscv.best_estimator_

