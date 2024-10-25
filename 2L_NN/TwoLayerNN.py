from torch import nn


class TwoLayerNN(nn.Module):
    def __init__(self, input_size):
        super(TwoLayerNN, self).__init__()

        # Input and Output Layer (Fully-Connected)
        self.layer1 = nn.Linear(input_size, 16)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.layer2(self.activation(self.layer1(x)))