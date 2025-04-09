import random
import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda
from torch.utils.data import DataLoader, random_split

SEED = 42
GENERATOR = torch.manual_seed(SEED)

DATASET_PATH = "./datasets/"
MODEL_PATH = "./models/"
MODEL_FILE = "model.pth"

def set_random_state():
    random.seed(SEED)
    np.random.seed(SEED)

def get_generator():
    return GENERATOR

def data_load(dataset_path):
    # Download training data from open datasets
    # Dataset stores the samples and their corresponding labels
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    target_transform = Lambda(lambda y: torch.zeros(
        10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1)
    )
    training_data = datasets.MNIST(
        root=dataset_path,
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform
    )

    test_data = datasets.MNIST(
        root=dataset_path,
        train=False,
        download=True,
        transform=transform,
    )

    return training_data, test_data

class ds_generator():
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.split_trset(num_clients)

    def split_trset(self, num_clients):
        assert(num_clients > 0)

        tr_set, _ = data_load(DATASET_PATH)

        portion = 1/num_clients
        fracs = list([ portion for i in range(num_clients) ])

        dsets = random_split(
            tr_set, fracs, GENERATOR
        )

        self.dsets = dsets

    def generate(self):
        #yield from self.dsets
        for ds in self.dsets:
            yield ds

    def get_trset(self):
        return next(self.generate())
    
def average_state_dicts(dicts):
    # Ensure the list is not empty
    if not dicts:
        return {}

    # Initialize a new dictionary
    avg_dict = {}

    # Iterate through keys of the first dictionary
    for key in dicts[0].keys():
        # Stack all tensors for this key
        stacked = torch.stack([d[key] for d in dicts])
        # Compute the mean along the 0th dimension
        avg_dict[key] = stacked.mean(dim=0)

    return avg_dict
