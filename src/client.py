import torch
from torch.utils.data import DataLoader

import os
os.environ["RAY_DEDUP_LOGS"]="0"
import ray

@ray.remote
class Client:
    def __init__(self, client_id, training=None):
        self.client_id = client_id
        self.model = None
        self.hp = None
        self.tr_set = training

    def receive_model(self, model, hp):
        self.model = model
        self.hp = hp

    def send_weights(self):
        return self.model.state_dict()

    def update_training(self, training):
        self.tr_set = training

    def local_train(self):
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), self.hp["lr"])

        # DataLoader wraps an iterable around the Dataset
        train_dataloader = DataLoader(
            self.tr_set, batch_size=self.hp["batch_size"], shuffle=True, num_workers=4,
        )

        self.model.train()
        for epoch in range(self.hp["epochs"]):
            print("Client %d - start epoch %d" % (self.client_id, epoch))
            running_loss = 0.0
            epoch_steps = 0

            for i, data in enumerate(train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_steps += 1

                if i % 100 == 0:  # print every 100 mini-batches
                    print(
                        "Client %d, epoch [%d, %5d] loss: %.3f"
                        % (self.client_id, epoch, i, running_loss / (epoch_steps+1))
                    )
                    running_loss = 0.0

        return self.send_weights()
