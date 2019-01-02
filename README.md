# PyScience
PyScience is a library that is aimed to help data scientists find the model that best fits the data. Although most of the features are already provided by the [scikit-learn package](https://scikit-learn.org/stable/), we introduce a more complete and compact set of utilities that include basic optimisation (achieved by means of [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)), scaling and introduction of dummy variables when needed. A basic summary with additional basic plots of the results is also included.

## Table of Contents  
- [Classification](#classification)
- [Regression](#regression)
- [Time series](#time-series)
- [How to use it](#how-to-use-it)

## Classification
For classification problems we provide
- random forest classification
- logistic regression
- nearest neighbours classification

Below is an example of output
```
----------------------------------------------------------------------------------------------------------------
                              executing kNN_classifier on file 'files/titanic.csv'                              
----------------------------------------------------------------------------------------------------------------
training kNN classifier...
best parameters are: 
         KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, metric='minkowski',
         metric_params=None, n_jobs=None, n_neighbors=11, p=2,
         weights='uniform')
best accuracy score is: 0.7338345864661654
              precision    recall  f1-score   support

           0       0.82      0.67      0.74       154
           1       0.47      0.68      0.56        68

   micro avg       0.67      0.67      0.67       222
   macro avg       0.65      0.67      0.65       222
weighted avg       0.72      0.67      0.68       222

```

and a plotted summary

![Alt text](gallery/cm.png?raw=true "cm")

## Regression
For regression problems we provide
- random forest regression
- linear regression
- nearest neighbours regression

Below is an example of output
```
----------------------------------------------------------------------------------------------------------------
                      executing random_forest_regression on file 'files/temperatures.csv'                       
----------------------------------------------------------------------------------------------------------------
training random forest regressor...
best parameters are: RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='log2', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
best accuracy score is: 0.8268988642336668
r2 score is: 0.8006997761077188
square root of residuals is: 5.096760087312774
```

and a plotted summary

![Alt text](gallery/regression_forest.png?raw=true  "regression_forest")

## Time series

- time series and ARIMA models

![Alt text](gallery/ARIMA.png?raw=true "ARIMA")

## How to use it
The code is written for python3.X only and the data must be in `pandas DataFrame` format. Example files are included in the `files` folder to run and test examples and to show what the data sets must look like.

The response variable column name must be `"actuals"` in the data file, whereas there is no restriction on the types and names for the predictors.

A `run.py` is provided: simply replace the name of the algorithm you want to run and the path to the data file you want to run it on. 