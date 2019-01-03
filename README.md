# PyScience
PyScience is a library that is aimed to help data scientists find the model that best fits the data. Although most of the features are already provided by the [scikit-learn package](https://scikit-learn.org/stable/), we introduce a more complete and compact set of utilities that include basic optimisation (achieved by means of [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)), scaling and introduction of dummy variables when needed. A basic summary with additional basic plots of the results is also included.

## Table of Contents  
- [Why shall I use it?](#Why-shall-I-use-it?)
- [How to use it](#how-to-use-it)
- [Classification](#classification)
- [Regression](#regression)
- [Time series](#time-series)


## Why shall I use it?
PyScience is *not* a substitute of any of the many `scikit-learn` implementations and algorithms (and links therein), which we still address the users to. On the other hand, though, we provide a ready-to-use out of the box collection of modules that 
- normalise the sources making them suitable for classification/regression by encoding, scaling and normalising the data 
- provide basic optimisation of the algorithms by means of cross-validation and parameters search
- offer pretty visualisations of the final results

Moreover, we provide the classes `BestClassifier` and `BestRegressor` that run through different algorithms, together with corresponding optimisation parameters, in order to determine the one that performs best *on the cross-validation data*. We output a ranking of algorithms by performance as established against [any metric](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) defined by the standard scikit-learn API. Notice that this does not necessarily imply that the chosen algorithm performs best on the *test data* as well. 


## How to use it
The code is written for python3.X only and the data must be in `pandas DataFrame` format. Example files are included in the `files` folder.

The column holding the response variable must be named `'actuals'`, whereas there is no restriction on the types and names for the predictors.

- `run_single_model.py`: runs a single algorithm as specified by user. Specify the model and the data source within the file.
- `run_best_classifier.py`: creates and instance of the `BestClassifier` class and runs through a list of specified models to determine the one of best fit on the validation data
- `run_best_regressor.py`: creates and instance of the `BestRegressor` class and runs through a list of specified models to determine the one of best fit on the validation data

Here is an example output:
```
-------------------------------------------------------------------------------------------------------------------------------------
                                                  evaluating RandomForestClassifier                                                  
-------------------------------------------------------------------------------------------------------------------------------------
best parameters are: RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=True, random_state=None, verbose=0, warm_start=False)
-------------------------------------------------------------------------------------------------------------------------------------
                                                   evaluating KNeighboursClassifier                                                  
-------------------------------------------------------------------------------------------------------------------------------------
best parameters are: KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=10, p=2,
           weights='distance')
-------------------------------------------------------------------------------------------------------------------------------------
                                                    evaluating LogisticRegression                                                    
-------------------------------------------------------------------------------------------------------------------------------------
best parameters are: LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='multinomial',
          n_jobs=None, penalty='l2', random_state=None, solver='sag',
          tol=0.0001, verbose=0, warm_start=False)

*************************************************************************************************************************************
                algorithm  f1_weighted
1   KNeighboursClassifier     0.982062
2      LogisticRegression     0.981981
0  RandomForestClassifier     0.936805
*************************************************************************************************************************************
```

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
