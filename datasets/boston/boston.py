import os
from util import pre_processing, load

'''
Fetches the data
Performs Exploratory Analysis (Datatypes, Correlation Matrix, Missing Values, Group Analysis, Descriptive Statistics)
Replaces column names to more meaningful ones if needed
Plots outliers
Returns a dataframe with zero n/a values and all columns
'''
def get_dataset():
    columns = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
    file_path = os.path.join(os.path.dirname(__file__), 'boston.csv')
    data = load.dataset(file_path, columns, None,1)
    # Drop n/a values
    data = pre_processing.clean_data(data)
    return data
