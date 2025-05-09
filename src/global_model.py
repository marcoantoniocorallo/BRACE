'''
    Definition of a simple classifier, found via model selection in classifier.py.
    The model is quite simple: a 1-hidden-layer-MLP with 512 units;
    The optimizer is Adam, batch-size is 50 and lr varies with the task.
'''

from utils import set_random_state, get_generator
import torch

# reproducibility
set_random_state()
GENERATOR = get_generator()

class MLPNet(torch.nn.Module):
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