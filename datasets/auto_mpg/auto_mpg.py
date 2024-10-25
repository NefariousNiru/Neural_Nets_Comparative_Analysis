import os.path
from util import pre_processing, load

'''
Fetches the data
Removes car names
Returns a dataframe with removed n/a values and all columns except car_name
'''
def get_dataset():
    columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']

    file_path = os.path.join(os.path.dirname(__file__), 'auto-mpg.data')
    data = load.dataset(file_path, columns, r'\s+', 0)

    # Drop car_name column
    data = data.drop(columns=["car_name"])

    # Drop n/a values
    data = pre_processing.clean_data(data)

    return data