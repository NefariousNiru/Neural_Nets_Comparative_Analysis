import os
from util import pre_processing, load

'''
Fetches the data
Performs Exploratory Analysis (Datatypes, Correlation Matrix, Missing Values, Group Analysis, Descriptive Statistics)
Replaces car names to car brands (more meaningful)
Plots outliers
Returns a dataframe with zero n/a values and all columns
'''
def get_dataset():
    # Define the relevant columns
    columns = ["Date", "Rented Bike Count", "Hour", "Temperature(C)", "Humidity(%)", "Wind speed (m/s)",
               "Visibility (10m)", "Dew point temperature(C)", "Solar Radiation (MJ/m2)", "Rainfall(mm)",
               "Snowfall (cm)", "Seasons", "Holiday", "Functioning Day"]

    file_path = os.path.join(os.path.dirname(__file__), 'SeoulBikeData.csv')
    data = load.dataset(file_path, columns, None, 1)

    # Drop n/a values
    data = pre_processing.clean_data(data)

    # One hot encoding
    data = pre_processing.encode_one_hot(['Seasons', 'Holiday', 'Functioning Day'], data)

    return data.drop(columns='Date')


