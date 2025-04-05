import ray
import numpy as np

# Start Ray
ray.init(ignore_reinit_error=True)

# ------------------------------
# Server Actor
# ------------------------------
@ray.remote
class Server:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.global_model = self.init_model()

    def init_model(self):
        # Initialize global model with random weights
        return np.random.rand(5)  # 5 weights for a simple model

    def send_model(self):
        return self.global_model

    def aggregate_weights(self, client_weights):
        # Average the weights from all clients
        return np.mean(client_weights, axis=0)

    def update_model(self, aggregated_weights):
        self.global_model = aggregated_weights

    def get_model(self):
        return self.global_model


# ------------------------------
# Client Actor
# ------------------------------
@ray.remote
class Client:
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = None
        self.local_data = self.generate_fake_data()

    def generate_fake_data(self):
        # Generate fake local data: X (features) and y (labels)
        X = np.random.rand(100, 5)  # 100 samples, 5 features
        y = X @ np.random.rand(5) + np.random.normal(0, 0.1, size=100)
        return X, y

    def receive_model(self, model_weights):
        self.model = model_weights

    def local_train(self, epochs=1, lr=0.1):
        # Very simple "training": one step of gradient descent
        X, y = self.local_data
        weights = self.model.copy()

        for _ in range(epochs):
            preds = X @ weights
            grad = X.T @ (preds - y) / len(X)
            weights -= lr * grad

        return weights


# ------------------------------
# Driver (Coordinator)
# ------------------------------
def federated_training(num_rounds=5, num_clients=3):
    server = Server.remote(num_clients)
    clients = [Client.remote(i) for i in range(num_clients)]

    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1} ---")

        # Server sends current model to clients
        global_weights = ray.get(server.send_model.remote())
        for client in clients:
            client.receive_model.remote(global_weights)

        # Clients perform local training
        client_weight_refs = [client.local_train.remote() for client in clients]
        client_weights = ray.get(client_weight_refs)

        # Server aggregates the client updates
        aggregated_weights = ray.get(server.aggregate_weights.remote(client_weights))
        server.update_model.remote(aggregated_weights)

        print(f"Aggregated model weights: {aggregated_weights.round(3)}")

    final_model = ray.get(server.get_model.remote())
    print("\n✅ Final global model weights:", final_model.round(3))


# ------------------------------
# Run the simulation
# ------------------------------
if __name__ == "__main__":
    federated_training()
    ray.shutdown()
