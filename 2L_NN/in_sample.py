import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datasets.auto_mpg import auto_mpg
from datasets.seoul_bike_sharing_demand import seoul_bike
from util import load, performance_metrics, pre_processing
from TwoLayerNN import TwoLayerNN


def run_auto_mpg_in_sample():
    data = auto_mpg.get_dataset()
    data = pre_processing.drop_outliers_inter_quartile(data)
    print(f"\nShape after removing Outliers using IQR, threshold 3, {data.shape}\n")

    X, y = load.get_x_y(data, 'mpg')
    X = X.astype(np.float32)

    input_size = X.shape[1]
    model = TwoLayerNN(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    X_tensor = torch.FloatTensor(X.values)
    y_tensor = torch.FloatTensor(y.values).view(-1, 1)

    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(X_tensor)  # Forward pass
        loss = criterion(outputs, y_tensor)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        if (epoch + 1) % 100 == 0:  # Print every 100 epochs
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model on the same dataset (in-sample evaluation)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor)  # Predictions
        in_sample_loss = criterion(y_pred, y_tensor)  # In-sample loss
        r2_score = performance_metrics.calculate_r2(y_tensor, y_pred)

    print(f'In-Sample Loss: {in_sample_loss.item()}')
    print(f'R2 Score: {r2_score}')

def run_seoul_bike_share_in_sample():
    data = seoul_bike.get_dataset()
    data = pre_processing.drop_outliers_inter_quartile(data)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")

    X, y = load.get_x_y(data, 'Rented Bike Count')
    X = X.astype(np.float32)

    input_size = X.shape[1]
    model = TwoLayerNN(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    X_tensor = torch.FloatTensor(X.values)
    y_tensor = torch.FloatTensor(y.values).view(-1, 1)

    num_epochs = 10000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(X_tensor)  # Forward pass
        loss = criterion(outputs, y_tensor)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        if (epoch + 1) % 1000 == 0:  # Print every 1000 epochs
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model on the same dataset (in-sample evaluation)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor)  # Predictions
        in_sample_loss = criterion(y_pred, y_tensor)  # In-sample loss
        r2_score = performance_metrics.calculate_r2(y_tensor, y_pred)

    print(f'In-Sample Loss: {in_sample_loss.item()}')
    print(f'R2 Score: {r2_score}')

if __name__ == "__main__":
    # run_auto_mpg_in_sample()
    run_seoul_bike_share_in_sample()