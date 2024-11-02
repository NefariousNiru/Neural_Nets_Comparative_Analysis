import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from Custom_NN.CustomNN import CustomNN
from util import performance_metrics

def train_and_evaluate(X_train, y_train, X_test, y_test, input_size):
    model = CustomNN(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 10000


    # Convert data to tensors
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        r2_score = performance_metrics.calculate_r2(y_test_tensor, y_pred)

    return r2_score