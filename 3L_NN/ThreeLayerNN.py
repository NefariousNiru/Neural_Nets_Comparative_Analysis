from torch import nn


class ThreeLayerNN(nn.Module):
    def __init__(self, input_size):
        super(ThreeLayerNN, self).__init__()

        # Input and Output Layer (Fully-Connected)
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.Linear(64, 1)
        )


    def forward(self, x):
        return self.network(x)