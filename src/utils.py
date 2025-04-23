import random
import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda
from torch.utils.data import DataLoader, random_split

import os
os.environ["RAY_DEDUP_LOGS"]="0"
from ray.util.queue import Queue

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

class ds_generator:
    def __init__(self, n_split):
        self.n_split = n_split
        self.queue = Queue()
        self._create_splits()

    def _create_splits(self):
        assert self.n_split > 0

        tr_set, _ = data_load(DATASET_PATH)

        portion = 1 / self.n_split
        fracs = [portion for _ in range(self.n_split)]

        dsets = random_split(tr_set, fracs, GENERATOR)

        for ds in dsets:
            self.queue.put(ds)

    def get_trset(self):
        return self.queue.get()


def average_state_dicts(dicts):
    if not dicts:
        return {}

    avg_dict = {}

    for key in dicts[0].keys():
        # Stack all tensors for this key
        stacked = torch.stack([d[key] for d in dicts])
        # Compute the mean along the 0th dimension
        avg_dict[key] = stacked.mean(dim=0)

    return avg_dict

def extract_percentage(l, percentage):
    n = int(len(l) * percentage)
    return random.sample(l, n)