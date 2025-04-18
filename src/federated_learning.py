from utils import set_random_state, get_generator, ds_generator, average_state_dicts
import os
import torch
import copy
from torch.utils.data import DataLoader

from global_model import MLPNet, HP

# reproducibility
set_random_state()
GENERATOR = get_generator()

# Start Ray
os.environ["RAY_DEDUP_LOGS"]="0"
import ray # imported after RAY_DEDUP_LOGS
ray.init(ignore_reinit_error=True)

@ray.remote
class Server:
    def __init__(self, model, hp,):
        self.global_model = model
        self.hp = hp

    def get_model(self):
        return self.global_model
    
    def get_hp(self):
        return self.hp

    def send_model(self):
        return copy.deepcopy(self.global_model)

    def aggregate_and_update(self, client_weights):
        avg = average_state_dicts(client_weights)
        self.global_model.load_state_dict(avg)

        return self.global_model.state_dict()

@ray.remote
class Client:
    def __init__(self, client_id, training):
        self.client_id = client_id
        self.model = None
        self.hp = None
        self.tr_set = training

    def receive_model(self, model, hp):
        self.model = model
        self.hp = hp

    def send_weights(self):
        return self.model.state_dict()

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

def federated_training(model, hp, num_rounds=1, num_clients=2):
    ds_gen = ds_generator(num_clients)

    server = Server.remote(model, hp)
    clients = [Client.remote(i, ds_gen.get_trset()) for i in range(num_clients)]

    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1} ---")

        # Server sends current model to clients
        global_model = ray.get(server.send_model.remote())

        for client in clients:
            client.receive_model.remote(global_model, hp)

        # Clients perform local training
        client_model_refs = [client.local_train.remote() for client in clients]
        client_models = ray.get(client_model_refs)

        # Server aggregates the client updates
        ray.get(server.aggregate_and_update.remote(client_models))

    final_model = ray.get(server.get_model.remote())
    print("\n✅ Final global model weights:", final_model)

# Run the simulation
if __name__ == "__main__":
    model = MLPNet()
    hp = HP
    federated_training(model = model, hp = HP)
    ray.shutdown()