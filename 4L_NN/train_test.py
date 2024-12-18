import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from datasets.auto_mpg import auto_mpg
from datasets.boston import boston
from datasets.seoul_bike_sharing_demand import seoul_bike
from util import load, performance_metrics, pre_processing, plot
from FourLayerNN import FourLayerNN

def run_auto_mpg():
    data = auto_mpg.get_dataset()
    data = pre_processing.drop_outliers_inter_quartile(data)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")

    X, y = load.get_x_y(data, 'mpg')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_size = X_train.shape[1]

    # Create an input layer with features from X as nodes
    model = FourLayerNN(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

    losses = []
    num_epochs = 10000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(X_train_tensor)  # Forward pass
        loss = criterion(outputs, y_train_tensor)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        losses.append(loss.item())
        if (epoch + 1) % 1000 == 0:  # Print every 10 epochs
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)  # Predictions
        test_loss = criterion(y_pred, y_test_tensor)  # Test loss
        r2_score = performance_metrics.calculate_r2(y_test_tensor, y_pred)

    print(f'Test Loss: {test_loss.item()}')
    print(f'R2 Score: {r2_score}')

    plot.plot_loss_vs_epoch(num_epochs, losses)
    plot.plot_feature_vs_pred(X_test, y_test, y_pred, "weight", 'mpg')
    plot.plot_feature_vs_pred(X_test, y_test, y_pred, "horsepower", 'mpg')

def run_seoul_bike_share():
    data = seoul_bike.get_dataset()
    data = pre_processing.drop_outliers_inter_quartile(data)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")

    X, y = load.get_x_y(data, 'Rented Bike Count')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    input_size = X_train.shape[1]
    model = FourLayerNN(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

    num_epochs = 10000
    losses = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(X_train_tensor)  # Forward pass
        loss = criterion(outputs, y_train_tensor)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        losses.append(loss.item())
        if (epoch + 1) % 1000 == 0:  # Print every 10 epochs
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)  # Predictions
        test_loss = criterion(y_pred, y_test_tensor)  # Test loss
        r2_score = performance_metrics.calculate_r2(y_test_tensor, y_pred)

    print(f'Test Loss: {test_loss.item()}')
    print(f'R2 Score: {r2_score}')

    plot.plot_loss_vs_epoch(num_epochs, losses)
    plot.plot_feature_vs_pred(X_test, y_test, y_pred, "Temperature(C)", 'Rented Bike Count')
    plot.plot_feature_vs_pred(X_test, y_test, y_pred, "Hour", 'Rented Bike Count')


def run_boston():
    data = boston.get_dataset()

    data = pre_processing.drop_outliers_zscore(data, 4)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")

    X, y = load.get_x_y(data, 'MEDV')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    input_size = X_train.shape[1]
    model = FourLayerNN(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

    num_epochs = 10000
    losses = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(X_train_tensor)  # Forward pass
        loss = criterion(outputs, y_train_tensor)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        losses.append(loss.item())
        if (epoch + 1) % 1000 == 0:  # Print every 10 epochs
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)  # Predictions
        test_loss = criterion(y_pred, y_test_tensor)  # Test loss
        r2_score = performance_metrics.calculate_r2(y_test_tensor, y_pred)

    print(f'Test Loss: {test_loss.item()}')
    print(f'R2 Score: {r2_score}')

    plot.plot_loss_vs_epoch(num_epochs, losses)
    plot.plot_feature_vs_pred(X_test, y_test, y_pred, "RM", 'MEDV')

if __name__ == "__main__":
    # run_auto_mpg()
    # run_seoul_bike_share()
    run_boston()