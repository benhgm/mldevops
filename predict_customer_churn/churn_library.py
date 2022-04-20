"""
<<Description of file>>

Author: Benjamin Ho
Last update: Apr 2022
"""

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

def import_data(pth):
    """
    returns a dataframe (df) for the csv file at pth

    Args:
        pth (str): path to the .csv data file

    Returns:
        df: pandas dataframe
    """
    return pd.read_csv("./data/bank_data.csv")