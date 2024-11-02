import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from datasets.auto_mpg import auto_mpg
from datasets.boston import boston
from datasets.seoul_bike_sharing_demand import seoul_bike
from util import load, performance_metrics, pre_processing
from FourLayerNN import FourLayerNN


def run_auto_mpg_cross_val(n_splits=5):
    data = auto_mpg.get_dataset()
    data = pre_processing.drop_outliers_inter_quartile(data)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")

    X, y = load.get_x_y(data, 'mpg')
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold = 1
    test_losses = []
    r2_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        input_size = X_train.shape[1]
        model = FourLayerNN(input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test.values)
        y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

        num_epochs = 10000
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            test_losses.append(loss.item())

        # Evaluate the model on this fold
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor)
            test_loss = criterion(y_pred, y_test_tensor)
            r2_score = performance_metrics.calculate_r2(y_test_tensor, y_pred)
            r2_scores.append(r2_score)

        print(f'Fold [{fold}/5], Loss: {test_loss.item()}, R2 Score: {r2_score}')
        fold += 1

    avg_test_loss = np.mean(test_losses)
    avg_r2_score = np.mean(r2_scores)
    print(f'\nAverage Test Loss: {avg_test_loss}')
    print(f'Average R2 Score: {avg_r2_score}')

def run_seoul_bike_share_cross_val(n_splits=5):
    data = seoul_bike.get_dataset()
    data = pre_processing.drop_outliers_inter_quartile(data)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")

    X, y = load.get_x_y(data, 'Rented Bike Count')
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold = 1
    test_losses = []
    r2_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index].astype(np.float32), X.iloc[test_index].astype(np.float32)
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        input_size = X_train.shape[1]
        model = FourLayerNN(input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test.values)
        y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

        num_epochs = 10000
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            test_losses.append(loss.item())
            optimizer.step()

        # Evaluate the model on this fold
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor)
            test_loss = criterion(y_pred, y_test_tensor)
            r2_score = performance_metrics.calculate_r2(y_test_tensor, y_pred)
            r2_scores.append(r2_score)

        print(f'Fold [{fold}/5], Loss: {test_loss.item()}, R2 Score: {r2_score}')
        fold += 1

    # Print average results
    avg_test_loss = np.mean(test_losses)
    avg_r2_score = np.mean(r2_scores)
    print(f'\nAverage Test Loss: {avg_test_loss}')
    print(f'Average R2 Score: {avg_r2_score}')

def run_boston_cross_validation():
    data = boston.get_dataset()

    # data = pre_processing.drop_outliers_zscore(data, 4)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")

    X, y = load.get_x_y(data, 'MEDV')
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Initialize K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold = 1
    r2_scores = []
    test_losses = []

    for train_index, test_index in kf.split(X):
        # Split data into training and test set for this fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test.values)
        y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

        # Initialize the model, loss, and optimizer
        input_size = X_train.shape[1]
        model = FourLayerNN(input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        num_epochs = 10000
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(X_train_tensor)  # Forward pass
            loss = criterion(outputs, y_train_tensor)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor)  # Predictions
            test_loss = criterion(y_pred, y_test_tensor)  # Test loss
            r2_score = performance_metrics.calculate_r2(y_test_tensor, y_pred)

            test_losses.append(test_loss.item())
            r2_scores.append(r2_score)

        print(f'Fold [{fold}/5], Loss: {test_loss.item()}, R2 Score: {r2_score}')
        fold += 1

    # Print average results
    avg_test_loss = np.mean(test_losses)
    avg_r2_score = np.mean(r2_scores)
    print(f'\nAverage Test Loss: {avg_test_loss}')
    print(f'Average R2 Score: {avg_r2_score}')

if __name__ == "__main__":
    # run_auto_mpg_cross_val()
    # run_seoul_bike_share_cross_val()
    run_boston_cross_validation()