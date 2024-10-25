from torch import nn


class FourLayerNN(nn.Module):
    def __init__(self, input_size):
        super(FourLayerNN, self).__init__()

        # Input and Output Layer (Fully-Connected)
        self.layer1 = nn.Linear(input_size, 64)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(32, 16)
        self.activation3 = nn.ReLU()
        self.output = nn.Linear(16, 1)


    def forward(self, x):
        return self.output(self.activation3(self.layer3(self.activation2(self.layer2(self.activation1(self.layer1(x)))))))