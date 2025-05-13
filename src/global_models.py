'''
    Definition of simple classifiers, found via model selection in classifier.py.
    The models are quite simple: 
        a 1-hidden-layer-MLP with 512 units for MNIST 
        a 2-layer CNN followed by 3 fully connected layers, 
            with batch normalization and dropout for FashionMNISt.
'''

from utils import set_random_state, get_generator
import torch
from torch import nn
import torch.nn.functional as F

# reproducibility
set_random_state()
GENERATOR = get_generator()

class MNISTModel(torch.nn.Module):
    def __init__(self, input_size = 28, hidden = 512, output_size = 10, generator = GENERATOR):
        super().__init__()

        # nn architecture
        self.input_size = input_size
        self.l1 = torch.nn.Linear(input_size * input_size, hidden)
        self.l2 = torch.nn.Linear(hidden, output_size)

        # initialize weights
        torch.nn.init.kaiming_uniform_(self.l1.weight, generator = generator)
        torch.nn.init.kaiming_uniform_(self.l2.weight, generator = generator)

    def forward(self, x: torch.Tensor):
        flatten = torch.nn.Flatten()
        x = flatten(x) # convert each 28x28 image into an array of 784 pixel values
        x = torch.relu(self.l1(x))
        x = torch.softmax(self.l2(x), 1)
        return x

class FashionMNISTModel(nn.Module):
    
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out

MNIST_HP = {
    "lr" : 0.00013292918943162168,
    "batch_size" : 50,
    "epochs" : 20,
}

FASHIONMNIST_HP = {
    "lr" : 0.000362561763457623,
    "batch_size" : 50,
    "epochs" : 20,
}

KMNIST_HP = {
    "lr" : 0.0006838478430964042,
    "batch_size" : 50,
    "epochs" : 20,
}