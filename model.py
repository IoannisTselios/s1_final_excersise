# model.py
from torch import nn
import torch

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super(MyAwesomeModel, self).__init__()
        # Your model architecture definition here
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu2 = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 12 * 12, 10)

    def forward(self, x):
        # Check if the input is a 4D tensor
        if x.ndim != 4:
            raise ValueError('Expected input to be a 4D tensor')

        # Check if each sample has shape [1, 28, 28]
        expected_shape = torch.Size([1, 28, 28])
        if x.shape[1:] != expected_shape:
            raise ValueError(f'Expected each sample to have shape {expected_shape}')

        # Rest of your forward logic here
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

# Use an instance of the class for convenience
myawesomemodel = MyAwesomeModel()
