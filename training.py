"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""
import pandas as pd
from sklearn import tree, linear_model
import joblib

def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    
    # Combine cleaned_df and outcome_df
    tree_regression = tree.DecisionTreeRegressor()
    columns = cleaned_df.columns[cleaned_df.columns != 'nomem_encr']
    tree_regression = tree_regression.fit(cleaned_df[columns], outcome_df)
    
    # Save the model
    joblib.dump(tree_regression, "model.joblib")

