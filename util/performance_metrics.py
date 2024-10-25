import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error


def calculate_r2(y_true, y_pred):
    # Convert tensors to NumPy arrays if necessary
    y_true = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.numpy() if isinstance(y_pred, torch.Tensor) else y_pred

    # Calculate the mean of the true values
    y_mean = np.mean(y_true)

    # Calculate SS_tot and SS_res
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Calculate R^2
    r2 = 1 - (ss_res / ss_tot)
    return r2

def get_mse(y_true, y_pred):
    return mean_squared_error(y_true=y_true, y_pred=y_pred)

def get_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))

def get_all(y_true, y_pred):
    mse = round(get_mse(y_true, y_pred), 5)
    r2 = round(get_r2_score(y_true, y_pred), 5)
    rmse = round(get_rmse(y_true, y_pred), 5)

    return {
        'MSE': f"{mse:.5f}",
        'R²': f"{r2:.5f}",
        'RMSE': rmse,
    }

def predvactual(y_test, y_test_pred, model_name):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=y_test, y=y_test_pred, color="orange")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='pink')
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title(f'{model_name} - Predicted vs Actual')
    plt.show()

