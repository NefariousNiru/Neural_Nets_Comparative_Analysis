from backward import backward_feature_selection
from feature_selection.backward import backward_feature_selection_categorical
from feature_selection.forward import forward_feature_selection_categorical
from forward import forward_feature_selection
from stepwise import stepwise_feature_selection
from datasets.auto_mpg import auto_mpg
from datasets.seoul_bike_sharing_demand import seoul_bike
from datasets.boston import boston
from util import pre_processing, load


def run_auto_fs():
    data = auto_mpg.get_dataset()
    data = pre_processing.drop_outliers_inter_quartile(data)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")
    X, y = load.get_x_y(data, 'mpg')
    selected = forward_feature_selection(X, y)

def run_auto_bs():
    data = auto_mpg.get_dataset()
    data = pre_processing.drop_outliers_inter_quartile(data)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")

    X, y = load.get_x_y(data, 'mpg')
    selected = backward_feature_selection(X, y)

def run_auto_sw():
    data = auto_mpg.get_dataset()
    data = pre_processing.drop_outliers_inter_quartile(data)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")

    X, y = load.get_x_y(data, 'mpg')
    selected = stepwise_feature_selection(X, y)

def run_seoul_fs():
    data = seoul_bike.get_dataset()
    data = pre_processing.drop_outliers_inter_quartile(data)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")
    X, y = load.get_x_y(data, 'Rented Bike Count')
    forward_feature_selection_categorical(X, y)

def run_seoul_bs():
    data = seoul_bike.get_dataset()
    data = pre_processing.drop_outliers_inter_quartile(data)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")
    X, y = load.get_x_y(data, 'Rented Bike Count')
    backward_feature_selection_categorical(X, y)


def run_seoul_sw():
    data = seoul_bike.get_dataset()
    data = pre_processing.drop_outliers_inter_quartile(data)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")
    X, y = load.get_x_y(data, 'Rented Bike Count')
    stepwise_feature_selection(X, y)


def run_boston_fs():
    data = boston.get_dataset()
    data = pre_processing.drop_outliers_zscore(data, 4)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")
    X, y = load.get_x_y(data, 'MEDV')
    forward_feature_selection(X, y)

def run_boston_bs():
    data = boston.get_dataset()
    data = pre_processing.drop_outliers_zscore(data, 4)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")
    X, y = load.get_x_y(data, 'MEDV')
    backward_feature_selection(X, y)

def run_boston_sw():
    data = boston.get_dataset()
    data = pre_processing.drop_outliers_zscore(data, 4)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")
    X, y = load.get_x_y(data, 'MEDV')
    stepwise_feature_selection(X, y)


def backward():
    # run_auto_bs()
    run_seoul_bs()
    # run_boston_bs()

def forward():
    # run_auto_fs()
    run_seoul_fs()
    # run_boston_fs()

def stepwise():
    # run_auto_sw()
    run_seoul_sw()
    # run_boston_sw()

if __name__ == "__main__":
    # forward()
    # backward()
    stepwise()






