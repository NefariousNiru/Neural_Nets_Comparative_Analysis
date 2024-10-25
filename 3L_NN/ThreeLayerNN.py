from torch import nn


class ThreeLayerNN(nn.Module):
    def __init__(self, input_size):
        super(ThreeLayerNN, self).__init__()

        # Input and Output Layer (Fully-Connected)
        self.layer1 = nn.Linear(input_size, 32)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(32, 16)
        self.activation2 = nn.ReLU()
        self.output = nn.Linear(16, 1)


    def forward(self, x):
        return self.output(self.activation2(self.layer2(self.activation1(self.layer1(x)))))