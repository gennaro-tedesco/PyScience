import shutil
columns = shutil.get_terminal_size().columns
import pandas as pd
from sklearn.model_selection import GridSearchCV

class BestRegressor:
    """
    class that fits a list of models to the training data to determine
    the model of best fit, where best fit is defined against any scoring_metric
    allowed by the scikit-learn API.

    models: a dictionary of {modelName: modelInstance()}
    params: a dictionary of {modelName: {modelParameters}}
    scoring_metric: ['r2', 'explained_variance', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    """
    def __init__(self, models, params, scoring_metric):
        if not isinstance(models, dict):
            raise ValueError("please specify a dictionary of {model_name: model_instance}")
        if not isinstance(params, dict):
            raise ValueError("please specify a dictionary of parameters")
        if set(models.keys())-set(params.keys()) != set():
            raise ValueError("No specified parameters for model(s) {}".format(set(models.keys())-set(params.keys())))

        self.models = models 
        self.params = params 
        self.single_classifier_best = {}
        self.scoring_metric = scoring_metric

    def fit(self, train_features, train_actuals):
        """
        fits the list of models to the training data, thereby obtaining in each case an evaluation score after GridSearchCV cross-validation
        """
        for name in self.models.keys():
            print('-'*shutil.get_terminal_size().columns)
            print("evaluating {}".format(name).center(columns))
            print('-'*shutil.get_terminal_size().columns)
            estimator = self.models[name]
            est_params = self.params[name]
            gscv = GridSearchCV(estimator, est_params, cv=5, scoring=self.scoring_metric)
            gscv.fit(train_actuals, train_features)
            print("best parameters are: {}".format(gscv.best_estimator_))
            self.single_classifier_best[name] = gscv
    
    def evaluation(self):
        """
        prints a summary report, ranking the models in terms of highest evaluation score
        """
        rows_list = []
        for name in self.single_classifier_best.keys():
            row = {}
            row['algorithm'] = name 
            row[self.scoring_metric] = self.single_classifier_best[name].best_score_
            rows_list.append(row)
        
        scoring_df = pd.DataFrame(rows_list)
        scoring_sorted = scoring_df.sort_values(self.scoring_metric, ascending=False)
        print()
        print('*'*shutil.get_terminal_size().columns)
        print(scoring_sorted)
        print('*'*shutil.get_terminal_size().columns)
        self.evaluation_scores = scoring_sorted