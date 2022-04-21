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

if __name__ == "__main__":
    test_import()
    test_eda()