import argparse
from utils import set_random_state, get_generator, ds_generator
import os

from global_model import MLPNet, HP
from server import Server
from client import Client

# reproducibility
set_random_state()
GENERATOR = get_generator()

# Start Ray
os.environ["RAY_DEDUP_LOGS"]="0"
import ray # imported after RAY_DEDUP_LOGS
ray.init(ignore_reinit_error=True)

def federated_training(model, hp, n_rounds=1, n_clients=5):
    ds_gen = ds_generator(n_clients * n_rounds)

    server = Server.remote(model, hp)
    clients = [Client.remote(i) for i in range(n_clients)]

    for round_num in range(n_rounds):
        print(f"\n--- Round {round_num + 1} ---")

        # Server sends current model to clients
        global_model = ray.get(server.send_model.remote())

        for client in clients:
            client.receive_model.remote(global_model, hp)
            client.update_training.remote(ds_gen.get_trset())

        # Clients perform local training
        client_model_refs = [client.local_train.remote() for client in clients]
        client_models = ray.get(client_model_refs)

        # Server aggregates the client updates
        ray.get(server.aggregate_and_update.remote(client_models))
    
    print("\n--- Training complete ---")
    accuracy = ray.get(server.test_model.remote())
    print("\nFinal global model accuracy:", accuracy)

# Run the simulation
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_clients', type=int, default=5)
    parser.add_argument('-r', '--rounds', type=int, default=2)
    n_clients = vars(parser.parse_args())['n_clients']
    n_rounds = vars(parser.parse_args())['rounds']

    model = MLPNet()
    hp = HP
    federated_training(model=model, hp=HP, n_clients=n_clients, n_rounds=n_rounds)
    ray.shutdown()