import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def process(data, column_nan_threshold, row_nan_threshold):
    data = data.copy()

    # Replace -5e+06 and 'MISSING' with NaN
    data.replace(-5e+06, np.nan, inplace=True)
    data.replace('MISSING', np.nan, inplace=True)

    # Remove rows where 'target' is 'UNKNOWN'
    data = data[data['target'] != 'UNKNOWN']

    # Create binary target variable (BAD = 1, GOOD = 0)
    data.insert(0, 'target_binary', (data['target'] == 'BAD').astype(int))

    # Remove unnecessary columns
    data.drop(columns=['id', 'decision_date', 'target'], inplace=True, errors='ignore')

    # Remove columns where more than 'column_nan_threshold' % of values are missing
    column_threshold = column_nan_threshold * data.shape[0]
    data = data.loc[:, data.isnull().sum() <= column_threshold]

    # Remove columns with the same value for all non-NaN observations
    data = data.loc[:, data.nunique(dropna=True) > 1]

    # Remove rows where more than 'row_nan_threshold' % of values are missing
    row_threshold = row_nan_threshold * data.shape[1]
    data = data[data.isnull().sum(axis=1) <= row_threshold]

    # split data into features (X) and target variable (y)
    X = data.drop(columns=['target_binary'])
    y = data['target_binary']

    return X, y





