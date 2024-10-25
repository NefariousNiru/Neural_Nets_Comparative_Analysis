from torch import nn


class CustomNN(nn.Module):
    def __init__(self, input_size):
        super(CustomNN, self).__init__()

        # Input and Output Layer (Fully-Connected)
        self.layer1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.activation1 = nn.ReLU()

        self.layer2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.activation2 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)

        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.activation3 = nn.ReLU()

        self.layer4 = nn.Linear(64, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.activation4 = nn.ReLU()

        self.output = nn.Linear(16, 1)


    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.activation1(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.dropout1(x)

        x = self.layer3(x)
        x = self.bn3(x)
        x = self.activation3(x)

        x = self.layer4(x)
        x = self.bn4(x)
        x = self.activation4(x)
        x = self.output(x)
        return x