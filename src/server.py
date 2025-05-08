from utils import average_state_dicts, data_load, DATASET_PATH
import torch
import copy

import os
os.environ["RAY_DEDUP_LOGS"]="0"
import ray

@ray.remote
class Server:
    def __init__(self, model, hp, task):
        self.model = model
        self.hp = hp
        self.task = task

    def get_model(self):
        return self.model
    
    def get_hp(self):
        return self.hp

    def send_model(self):
        return copy.deepcopy(self.model)

    def aggregate_and_update(self, client_weights):
        avg = average_state_dicts(client_weights)
        self.model.load_state_dict(avg)

        return self.model.state_dict()
    
    def test_model(self):
        _, testset = data_load(DATASET_PATH, dataset_name=self.task)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=4, shuffle=False, num_workers=2
        )

        correct = 0
        total = 0

        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.eval()

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total