import argparse
from utils import set_random_state, get_generator, ds_generator, extract_percentage
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

def federated_training(model, hp, n_rounds=1, n_clients=5, percentage=1, rtime=False):
    n_split = int(n_clients * percentage) * (n_rounds if rtime else 1)
    ds_gen = ds_generator(n_split)

    server = Server.remote(model, hp)
    clients = [Client.remote(i, ds_gen) for i in range(n_clients)]

    for round_num in range(n_rounds):
        print(f"\n--- Round {round_num + 1} ---")

        # Server sends current model to some clients
        global_model = ray.get(server.send_model.remote())

        chosen_clients = extract_percentage(clients, percentage)
        for client in chosen_clients:
            client.receive_model.remote(global_model, hp)
            if rtime:
                client.update_training.remote()

        # Clients perform local training
        client_model_refs = [client.local_train.remote() for client in chosen_clients]
        client_models = ray.get(client_model_refs)

        # Server aggregates the client updates
        ray.get(server.aggregate_and_update.remote(client_models))
    
    print("\n--- Training complete ---")
    accuracy = ray.get(server.test_model.remote())
    print("\nFinal global model accuracy:", accuracy)

# Run the simulation
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                "-n     | --n_clients: number of clients (default: 5)\n"
                "-r     | --rounds: number of rounds (default: 2)\n"
                "-p     | --percentage: percentage of clients to use (default: 1 (all))\n"
                "-rtime | --realtime: different tr.set at each round \n"
                "Ex:\n"
                "python federated_learning.py -n 10 -r 5 -p 0.5 \n"
                "will run the simulation with 10 clients, 5 rounds, "
                "50% of clients participating in each round.\n"
                "\n"
    )
    parser.add_argument('-n', '--n_clients', type=int, default=5)
    parser.add_argument('-r', '--rounds', type=int, default=2)
    parser.add_argument('-p', '--percentage', type=float, default=1.0)
    parser.add_argument('-rtime', '--realtime', action='store_true', default=False)
    n_clients = vars(parser.parse_args())['n_clients']
    n_rounds = vars(parser.parse_args())['rounds']
    percentage = vars(parser.parse_args())['percentage']
    rtime = vars(parser.parse_args())['realtime']

    model = MLPNet()
    hp = HP
    federated_training(
        model=model, 
        hp=HP, 
        n_clients=n_clients, 
        n_rounds=n_rounds, 
        percentage=percentage,
        rtime=rtime
    )
    ray.shutdown()