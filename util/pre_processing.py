import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def print_dtypes(dataset_name, data):
    print()
    print(f"{dataset_name} dataset datatypes")
    print(data.dtypes)

def clean_data(data):
    shape_with_null_values = data.shape
    data = data.dropna()
    shape_without_null_values = data.shape
    print()
    print(f"Shape with n/a values: {shape_with_null_values}")
    print(f"Shape without n/a values: {shape_without_null_values}")
    print(f"n/a values: {shape_with_null_values[0] - shape_without_null_values[0]}")
    return data

def encode_one_hot(columns, data):
    # One-hot encode the specified columns
    # drop_first=True to avoid multi collinearity
    print()
    print(f"One hot encoding {columns}")
    data = pd.get_dummies(data, columns=columns, drop_first=True)
    return data

def encode_label(column, data):
    pass

def scale(data):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    # Scale only the numerical columns
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data


def drop_outliers_inter_quartile(data):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

    for column in numeric_columns:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter the data to remove outliers
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    return data

def drop_outliers_zscore(data, threshold=3):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

    for column in numeric_columns:
        # Calculate the Z-score for each value in the column
        col_mean = data[column].mean()
        col_std = data[column].std()
        z_scores = (data[column] - col_mean) / col_std

        # Filter the data to remove rows where Z-score is above the threshold
        data = data[np.abs(z_scores) <= threshold]

    return data