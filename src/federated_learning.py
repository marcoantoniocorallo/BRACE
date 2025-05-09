'''
    Implementation of a Federated Learning simulation using Ray.
    The system takes as parameters the model and hyperparameters to use, as well as:
        - the total number of clients (benign and byzantine), 
        - the number of rounds,
        - the percentage of clients to use in each round, 
        - the number of byzantine clients,
        - and whether to use a different training set at each round (real-time task).

    The simulation ran on different configurations of these parameters, 
    using the model and hyperparameters found by the model selection in mnist_classifier.py.

    Results are reported in the report.
'''

import argparse
from utils import set_random_state, get_generator, ds_generator, extract_percentage
import os

from global_model import MLPNet, MNIST_HP, FASHIONMNIST_HP
from server import Server
from client import Ray_Client as Client
from byzantineClient import ByzantineClient

# reproducibility
set_random_state()
GENERATOR = get_generator()

# Start Ray
os.environ["RAY_DEDUP_LOGS"]="0"
import ray # imported after RAY_DEDUP_LOGS
ray.init(ignore_reinit_error=True)

def federated_training(
        model, 
        hp,
        task,
        n_rounds=1, 
        n_clients=5, 
        percentage=1, 
        n_byzantine=1, 
        rtime=False
    ):
    ds_gen = ds_generator(n_clients=n_clients, n_rounds=n_rounds, rtime=rtime, task=task)

    server = Server.remote(model, hp, task=task)
    n_benign = n_clients - n_byzantine
    clients = [Client.remote(i, ds_gen, task=task) for i in range(n_benign)] + \
                [ByzantineClient.remote(i, ds_gen, task=task) for i in range(n_benign, n_clients)]

    for round_num in range(n_rounds):
        print(f"\n--- Round {round_num + 1} ---")

        # Server sends current model to some clients
        global_model = ray.get(server.send_model.remote())

        chosen_clients = extract_percentage(clients, percentage)
        for client in chosen_clients:
            client.receive_model.remote(global_model, hp)
            
            # if real-time task, change the dataset at each round
            if rtime and round_num > 0:
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
                "-t     | --task: task to run (mnist | fashionmnist)\n"
                "-n     | --n_clients: number of clients (default: 5)\n"
                "-r     | --rounds: number of rounds (default: 2)\n"
                "-p     | --percentage: percentage of clients to use (default: 1 (all))\n"
                "-b     | --byzantine: number of byzantine clients (default: 1)\n"
                "-rtime | --realtime: different tr.set at each round \n"
                "Ex:\n"
                "python federated_learning.py -n 10 -r 5 -p 0.5 \n"
                "will run the simulation with 10 clients, 5 rounds, "
                "50% of clients participating in each round.\n"
                "\n"
    )
    parser.add_argument('-t', '--task', type=str)
    parser.add_argument('-n', '--n_clients', type=int, default=5)
    parser.add_argument('-r', '--rounds', type=int, default=2)
    parser.add_argument('-p', '--percentage', type=float, default=1.0)
    parser.add_argument('-b', '--byzantine', type=int, default=1)
    parser.add_argument('-rtime', '--realtime', action='store_true', default=False)
    task = vars(parser.parse_args())['task']
    n_clients = vars(parser.parse_args())['n_clients']
    n_rounds = vars(parser.parse_args())['rounds']
    percentage = vars(parser.parse_args())['percentage']
    rtime = vars(parser.parse_args())['realtime']
    n_byzantine = vars(parser.parse_args())['byzantine']

    # validate inputs
    assert(task in ["mnist", "fashionmnist", "fashion"]), "Task must be mnist or fashionmnist"
    assert(n_clients > 0), "Number of clients must be greater than 0"
    assert(n_rounds > 0), "Number of rounds must be greater than 0"
    assert( 0 < percentage <= 1), "Percentage of clients must be between 0 and 1"
    assert(n_clients >= n_byzantine), "Number of clients must be greater than number of byzantine clients"

    model = MLPNet()
    hp = MNIST_HP if task == "mnist" else FASHIONMNIST_HP
    federated_training(
        model=model, 
        hp=hp,
        task=task,
        n_clients=n_clients, 
        n_rounds=n_rounds, 
        percentage=percentage,
        rtime=rtime,
        n_byzantine=n_byzantine
    )
    ray.shutdown()