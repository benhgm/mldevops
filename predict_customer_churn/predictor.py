"""
Module for making predictions about customer churn

Author: Benjamin Ho
Last update: Apr 2022
"""
import json
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import churn_library as cl


class Predictor():
    """
    Predictor class for making predictions about customer churn
    """

    def __init__(self, cfg, is_train=True):
        super(Predictor, self).__init__()

        # init configuration dictionary
        self.cfg = cfg
        self.is_train = is_train

        # define input/output folder paths
        self.images_path = self.cfg['images_path']
        self.data_path = self.cfg['data_path']
        self.log_path = self.cfg['log_path']

        # define prediction and model params
        self.cat_columns = self.cfg['category_columns']
        self.cols_to_keep = self.cfg['columns_to_keep']
        self.param_grid = self.cfg['param_grid']

        # init models
        self.cv_rfc = GridSearchCV(
            estimator=RandomForestClassifier(
                random_state=42),
            param_grid=self.param_grid,
            cv=5)
        self.lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    def prepare_data(self):
        """
        Function to prepare dataset for training
        """
        # get data
        self.data = {}
        self.data['raw_input'] = cl.import_data(self.data_path)

        # perform eda
        cl.perform_eda(self.data['raw_input'])

        # encode categorical data
        encoder_input = self.data['raw_input']
        self.data['encoded'] = cl.encoder_helper(encoder_input,
                                                 self.cat_columns)

        # perform feature engineering
        model_inputs = cl.perform_feature_engineering(
            self.data['encoded'], self.cols_to_keep
        )
        self.data['X_filtered'] = model_inputs[0]
        self.data['X_train'] = model_inputs[1]
        self.data['X_test'] = model_inputs[2]
        self.data['y_train'] = model_inputs[3]
        self.data['y_test'] = model_inputs[4]

    def train_model(self):
        """
        Function to train a classification model
        """
        # init a Random Forest Classifier
        self.cv_rfc.fit(self.data['X_train'], self.data['y_train'])

        # init a Logistic Regressor
        self.lrc.fit(self.data['X_train'], self.data['y_train'])

        # store predictions
        self.preds = {}
        self.preds["y_train_rf"] = self.cv_rfc.best_estimator_.predict(
            self.data['X_train'])
        self.preds["y_test_rf"] = self.cv_rfc.best_estimator_.predict(
            self.data['X_test'])
        self.preds["y_train_lr"] = self.lrc.predict(self.data['y_train'])
        self.preds["y_test_lr"] = self.lrc.predict(self.data['y_test'])

    def post_process_results(self):
        """
        Function to post process results
        - Plot relevant curves
        - Save trained model
        """
        # generate classification report
        cl.classification_report(self.data['y_train'],
                                 self.data['y_test'],
                                 self.preds)

        # generate feature importance plots
        importances = self.cv_rfc.best_estimator_.feature_importances_
        output_path = self.images_path + '/feature_importances.jpg'
        cl.feature_importance_plot(importances,
                                   self.data['X_filtered'],
                                   output_path)

        # plot roc curves
        roc_output = self.images_path + '/roc_curves.jpg'
        cl.plot_roc_curves(self.lrc,
                           self.cv_rfc,
                           self.data['X_test'],
                           self.data['y_test'],
                           roc_output)

        # save models
        joblib.dump(self.cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(self.lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    with open('./config.json', 'r') as file:
        config = json.load(file)
    predictor = Predictor(config)
    predictor.prepare_data()
    predictor.train_model()
    predictor.post_process_results()
