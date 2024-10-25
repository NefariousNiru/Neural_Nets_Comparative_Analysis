import pandas as pd


def dataset(file_path: str, column_names: list, sep=None, skiprows=0):
    if sep:
        data = pd.read_csv(file_path, sep=sep, names=column_names, na_values='?', skiprows=skiprows)
    else:
        data = pd.read_csv(file_path, names=column_names, na_values='?', skiprows=skiprows)

    return data

def get_x_y(data, y_column):
    return data.drop(columns=y_column), data[y_column]