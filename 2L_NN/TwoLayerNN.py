from torch import nn


class TwoLayerNN(nn.Module):
    def __init__(self, input_size):
        super(TwoLayerNN, self).__init__()

        # Input and Output Layer (Fully-Connected)
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x)