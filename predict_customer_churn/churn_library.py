"""
<<Description of file>>

Author: Benjamin Ho
Last update: Apr 2022
"""

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth):
    """
    returns a dataframe (df) for the csv file at pth

    Args:
        pth (str): path to the .csv data file

    Returns:
        df: pandas dataframe
    """
    return pd.read_csv("./data/bank_data.csv")


def perform_eda(df):
    # visualize data
    print(df.head())
    print("The size of the data is: ", df.shape)
    print(
        "There are {} categories and {} data rows.".format(
            df.shape[1],
            df.shape[0]))

    # check if there is any empty data
    print(df.isnull().sum())
