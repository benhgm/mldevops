"""
Library of functions for predicting customer churn

Author: Benjamin Ho
Last update: Apr 2022
"""

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import train_test_split
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
    return pd.read_csv(pth)


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
    _, axs = plt.subplots(2, 1, figsize=(20, 10))

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

    Returns:
        df: pandas dataframe
    """
    for category in category_list:
        new_category_list = []
        category_groups = df.groupby(category).mean()['Churn']
        for val in df[category]:
            new_category_list.append(category_groups.loc[val])
        new_category_name = category + '_Churn'
        df[new_category_name] = new_category_list
    return df


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
    return X, X_train, X_test, y_train, y_test


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


def feature_importance_plot(importances, X_data, output_pth):
    """
    Function to generate a feature importance plot for a
    random forest classifier

    Args:
        importances: a RandomoForestClassfier model feature_importances_ object
        X_data: pandas dataframe of X values
        output_pth (str): path to save the plot
    """
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def plot_roc_curves(lrc_model, rfc_model, X_test, y_test, output_path):
    """
    Function to plot the roc curves

    Args:
        lrc_model: scikit-image LogisticRegression model object
        rfc_model (): scikit-image RandomForesClassifier model object
        X_test (arr): array of test inputs
        y_test (arr): array of test outputs
        output_path (str): output path to save the roc plot
    """
    lrc_plot = plot_roc_curve(lrc_model, X_test, y_test)

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        rfc_model.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(output_path)
