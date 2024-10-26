import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from datasets.auto_mpg import auto_mpg
from util import load, performance_metrics, pre_processing
from TwoLayerNN import TwoLayerNN
from datasets.seoul_bike_sharing_demand import seoul_bike


def run_auto_mpg_cross_val():
    data = auto_mpg.get_dataset()
    data = pre_processing.drop_outliers_inter_quartile(data)
    print(f"\nShape after removing Outliers using IQR, threshold 3, {data.shape}\n")

    X, y = load.get_x_y(data, 'mpg')
    X = X.astype(np.float32).values
    y = y.values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    r2_scores = []
    losses = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        input_size = X_train.shape[1]
        model = TwoLayerNN(input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

        num_epochs = 10000
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(X_train_tensor)  # Forward pass
            loss = criterion(outputs, y_train_tensor)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor)  # Predictions
            test_loss = criterion(y_pred, y_test_tensor)  # Test loss
            r2_score = performance_metrics.calculate_r2(y_test_tensor, y_pred)

        print(f'Fold [{fold}/5], Loss: {test_loss.item()}, R2 Score: {r2_score}')
        r2_scores.append(r2_score)
        losses.append(test_loss.item())
        fold += 1

    print(f'\nAverage Test Loss: {np.mean(losses)}')
    print(f'Average R2 Score: {np.mean(r2_scores)}')


def run_seoul_bike_share_cross_val():
    data = seoul_bike.get_dataset()
    data = pre_processing.drop_outliers_inter_quartile(data)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")

    X, y = load.get_x_y(data, 'Rented Bike Count')
    X = X.astype(np.float32).values
    y = y.values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    r2_scores = []
    losses = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        input_size = X_train.shape[1]
        model = TwoLayerNN(input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

        num_epochs = 10000
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(X_train_tensor)  # Forward pass
            loss = criterion(outputs, y_train_tensor)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor)  # Predictions
            test_loss = criterion(y_pred, y_test_tensor)  # Test loss
            r2_score = performance_metrics.calculate_r2(y_test_tensor, y_pred)

        print(f'Fold [{fold}/5], Loss: {test_loss.item()}, R2 Score: {r2_score}')
        r2_scores.append(r2_score)
        losses.append(test_loss.item())
        fold += 1

    print(f'\nAverage Test Loss: {np.mean(losses)}')
    print(f'Average R2 Score: {np.mean(r2_scores)}')


if __name__ == "__main__":
    # run_auto_mpg_cross_val()
    run_seoul_bike_share_cross_val()