'''
    This module implements a byzantine client for federated learning.

    This kind of byzantine client aim to launch a "Data Poisoning Attack", that is:
    the attacker cannot directly modify the local model parameters but can only falsify
    or tamper the training data.

    In particular, it implements a "Static Label Flipping" by flipping 
    the label of the digit image of 1 to 7. It is considered the default attack method.
'''

from client import Client
import torch
import os

from flippingSubset import FlippingSubset
os.environ["RAY_DEDUP_LOGS"]="0"
import ray

tensor_one = torch.zeros(10, dtype=torch.float)
tensor_seven = torch.zeros(10, dtype=torch.float)
tensor_nine = torch.zeros(10, dtype=torch.float)
tensor_one[1] = 1.0
tensor_seven[7] = 1.0
tensor_nine[9] = 1.0

@ray.remote
class ByzantineClient(Client):
    def __init__(self, client_id, task, training=None):
        Client.__init__(self, client_id, task, training)
        self.tr_set = FlippingSubset(self.tr_set)
        self.label_flip()

    def label_flip(self):
        for i, (data, label) in enumerate(self.tr_set):

            if self.task == "mnist": # flip digits 1 and 7
                if torch.equal(label, tensor_one):
                    self.tr_set[i] = (data, 7)
                elif torch.equal(label, tensor_seven):
                    self.tr_set[i] = (data, 1)

            elif self.task == "fashionmnist": # flip sneakers and ankle boots (7 and 9)
                if torch.equal(label, tensor_seven):
                    self.tr_set[i] = (data, 9)
                elif torch.equal(label, tensor_nine):
                    self.tr_set[i] = (data, 7)
        print(f"ByzantineClient {self.client_id} - flipped labels")