"""
Script for testing functions in churn_library.py and performing logging

Author: Benjamin Ho
Last Updated On: Apr 2022
"""

import os
import logging
import churn_library as churn

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

DF = churn.import_data("./data/bank_data.csv")

CAT_COLUMNS = ['Gender',
               'Education_Level',
               'Marital_Status',
               'Income_Category',
               'Card_Category']

COLS_TO_KEEP = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
    'Income_Category_Churn', 'Card_Category_Churn'
]


def test_import():
    '''
    test data import
    '''
    try:
        df = churn.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test eda function
    '''
    try:
        churn.perform_eda(DF)
        assert os.path.isfile('./images/eda_bivariate_plot.jpg')
        assert os.path.isfile('./images/eda_univariate_plots.jpg')
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: File does not exist"
        )
        raise err


def test_encoder_helper():
    '''
    test the encoder helper function
    '''
    try:
        num_columns_before = DF.shape[1]
        churn.encoder_helper(DF, CAT_COLUMNS)
        num_columns_after = DF.shape[1]
        assert num_columns_after - num_columns_before == 5
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: Some categorical data were not encoded"
        )
        raise err


def test_perform_feature_engineering():
    '''
    test the feature engineering function
    '''
    try:
        X_train, X_test, y_train, y_test = churn.perform_feature_engineering(
            DF, COLS_TO_KEEP)
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The training and test sets have different shapes."
        )
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
