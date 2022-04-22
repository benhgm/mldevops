"""
<<Description of file>>

Author: Benjamin Ho
Last update: Apr 2022
"""

from random import Random
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
    Returns a dataframe (df) for the csv file at pth

    Args:
        pth (str): path to the .csv data file

    Returns:
        df: pandas dataframe
    """
    return pd.read_csv("./data/bank_data.csv")


def perform_eda(df):
    """
    Function for performing Explorative Data Analysis (EDA)
    Function does the following:
    1. Visualizes input dataframe
    2. Checks for missing data and alerts user
    3. Displays various plots for analysis

    Args:
        df (pandas dataframe): pandas dataframe object
    """
    # visualize data
    print(df.head())
    print("The size of the data is: ", df.shape)
    print(
        "There are {} categories and {} data rows.".format(
            df.shape[1],
            df.shape[0]))

    # check if there is any missing numbered data
    is_null_table = df.isnull().sum()
    print(is_null_table)
    df.info()
    assert is_null_table.all(
    ) == 0, "There are missing values in the data. Please clean up missing data before continuing."

    # visualize general statistics
    print(df.describe())

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Initialize figure
    fig, axs = plt.subplots(2, 1, figsize=(20, 10))

    # Visualize univariate plots
    axs[0].set_title('Customer Age')
    df['Customer_Age'].hist(ax=axs[0])
    axs[1].set_title('Marital Status')
    df.Marital_Status.value_counts('normalize').plot(kind='bar', ax=axs[1])
    plt.savefig('./images/eda_univariate_plots.jpg')
    # plt.show()
    plt.close()

    # Visualize bivariate plot
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title('Heat Map of Bivariate Relationships')
    plt.savefig('./images/eda_bivariate_plot.jpg')
    # plt.show()


def encoder_helper(df, category_list):
    """
    Function to encode each categorical column into a new column representing
    the proportion of total churn

    Args:
        df: pandas dataframe
        category_list: list of names of columns with categorical data
    """
    for category in category_list:
        new_category_list = []
        category_groups = df.groupby(category).mean()['Churn']
        for val in df[category]:
            new_category_list.append(category_groups.loc[val])
        new_category_name = category + '_Churn'
        df[new_category_name] = new_category_list


def perform_feature_engineering(df, cols_to_keep):
    """
    Function to split the dataframe into training and test sets

    Args:
        df: pandas dataframe
        cols_to_keep: list of columns to retain as features for the model

    Returns:
        X_train (arr): Array of training features
        X_test (arr): Array of test features
        y_train (arr): Array of training outputs
        y_test (arr): Array of test outputs
    """
    y = df['Churn']
    X = pd.DataFrame()
    X[cols_to_keep] = df[cols_to_keep]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                predictions):
    """
    Function to generate a scikit-image classification report and save it as an image

    Args:
        y_train (arr): array of training outputs
        y_test (arr): array of test outputs
        predictions (arr): array of predictions from the classifier
    """
    for key, _ in predictions.items():
        if "test" in key:
            print("Classification report for", key)
            results = classification_report(y_test, predictions[key])
            print(results)
            make_text_plot(key, results)
        if "train" in key:
            print("Classification report for", key)
            results = classification_report(y_train, predictions[key])
            print(results)
            make_text_plot(key, results)


def make_text_plot(title, text, show=True):
    """
    Creates a plot for a skimage classification report and saves it as a jpg image

    Args:
        title (str): Title of the plot
        text: scikit-image classification report
        show (bool, optional): whether to show the plot or not. Defaults to True.
    """
    text_kwargs = dict(ha='center', va='center', fontsize=20)
    title_font = {'family': 'serif',
                  'color': 'darkred',
                  'weight': 'bold',
                  'size': 24}
    plt.subplots()
    plt.text(0.5, 0.5, text, **text_kwargs)
    plt.title(title, fontdict=title_font)
    plt.axis('off')
    plt.savefig('./images/' + title + '.jpg')
    if show:
        plt.show()
        plt.close()


if __name__ == "__main__":
    path = "./data/bank_data.csv"
    df = import_data(path)
    perform_eda(df)

    # group columns into categories
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    encoder_helper(df, cat_columns)

    cols_to_keep = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]

    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df, cols_to_keep)

    # init a RandomForestClassifier
    rfc = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # init a Logistic Regressor
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(X_train, y_train)

    preds = {}
    preds["y_train_rf"] = cv_rfc.best_estimator_.predict(X_train)
    preds["y_test_rf"] = cv_rfc.best_estimator_.predict(X_test)
    preds["y_train_lr"] = lrc.predict(X_train)
    preds["y_test_lr"] = lrc.predict(X_test)

    classification_report_image(y_train, y_test, preds)

    # quant_columns = ['Customer_Age',
    #                  'Dependent_count',
    #                  'Months_on_book',
    #                  'Total_Relationship_Count',
    #                  'Months_Inactive_12_mon',
    #                  'Contacts_Count_12_mon',
    #                  'Credit_Limit',
    #                  'Total_Revolving_Bal',
    #                  'Avg_Open_To_Buy',
    #                  'Total_Amt_Chng_Q4_Q1',
    #                  'Total_Trans_Amt',
    #                  'Total_Trans_Ct',
    #                  'Total_Ct_Chng_Q4_Q1',
    #                  'Avg_Utilization_Ratio']
