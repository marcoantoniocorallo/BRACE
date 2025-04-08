import argparse
from itertools import product
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

DATASET_PATH = "./datasets/"
MODEL_PATH = "./models/"
MODEL_FILE = "model.pth"
SEED = 42

class MLPNet(torch.nn.Module):
    def __init__(self, input_size = 28, hidden = 512, output_size = 10, seed = 42):
        super().__init__()
        
        # set seed
        gen = torch.Generator().manual_seed(seed)

        # define nn architecture
        self.input_size = input_size
        self.l1 = torch.nn.Linear(input_size * input_size, hidden)
        self.l2 = torch.nn.Linear(hidden, output_size)

        # initialize weights
        torch.nn.init.kaiming_uniform_(self.l1.weight, generator = gen)
        torch.nn.init.kaiming_uniform_(self.l2.weight, generator = gen)

    def forward(self, x: torch.Tensor):
        flatten = nn.Flatten()
        x = flatten(x) # convert each 28x28 image into an array of 784 pixel values
        x = torch.relu(self.l1(x))
        x = torch.softmax(self.l2(x), 1)
        return x

def train_model(config):
    dir_path = DATASET_PATH

    model = MLPNet(hidden = config["hidden"])
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"])
    loss_fn = nn.CrossEntropyLoss() # suitable for multiclass classification tasks like MNIST

    trainset, _ = data_load(dir_path)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    # DataLoader wraps an iterable around the Dataset
    train_dataloader = DataLoader(
        train_subset, batch_size=config["batch_size"], shuffle=True, num_workers=4,
    )
    val_dataloader = DataLoader(
        val_subset, batch_size=config["batch_size"], shuffle=True, num_workers=4,
    )

    for epoch in range(config["epochs"]):
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.train()
        print(f"Epoch {epoch + 1}\n-------------------------------")
        running_loss = 0.0
        epoch_steps = 0


        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1

            if i % 100 == 0:  # print every 100 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / (epoch_steps+1))
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_dataloader, 0):
                inputs, labels = data

                outputs = model(inputs)
                _, predicted_label = torch.max(outputs.data, 1)
                _, true_label = torch.max(labels.data, 1)
                total += labels.size(0)

                correct += (predicted_label == true_label).sum().item()

                loss = loss_fn(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        train.report(
            {"val_loss": val_loss / val_steps, "accuracy": correct / total}
        )

    print("Finished Training")

def test_model(model, dir_path):
    _, testset = data_load(dir_path)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def save_model(model):
    torch.save(model.state_dict(), MODEL_PATH + MODEL_FILE)
    print("Saved PyTorch Model State to " + MODEL_PATH + MODEL_FILE)

def load_model(model):
    model.load_state_dict(torch.load(MODEL_PATH + MODEL_FILE, weights_only=True))
    return model

# retrain on the entire training set!
def retrain(config):
    dir_path = DATASET_PATH
    model = MLPNet(hidden = config["hidden"])
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"])

    loss_fn = nn.CrossEntropyLoss()

    trainset, _ = data_load(dir_path) # full trset

    # DataLoader wraps an iterable around the Dataset
    train_dataloader = DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True, num_workers=4,
    )

    model.train()
    for epoch in range(config["epochs"]):
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        print(f"Epoch {epoch + 1}\n-------------------------------")
        running_loss = 0.0
        epoch_steps = 0

        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1

            if i % 100 == 0:  # print every 100 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / (epoch_steps+1))
                )
                running_loss = 0.0

    save_model(model)

def predict(model, data, classes):
    classes = [
        "Zero",
        "One",
        "Two",
        "Three",
        "Four",
        "Five",
        "Six",
        "Seven",
        "Eight",
        "Nine",
    ]

    model.eval()
    
    with torch.no_grad():
        for i in range(10):
            x, y = data[i][0], data[i][1]
            pred = model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')

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

def model_selection(config):
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=20,
        grace_period=1,
        reduction_factor=2,
    )

    result = tune.run(
        train_model,
        resources_per_trial={"cpu": 2},
        config=config,
        num_samples= 8,
        scheduler=scheduler,
        
    )

    best_trial = result.get_best_trial("val_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['val_loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    return best_trial

def main():
    parser = argparse.ArgumentParser(description='--training (-t) for model selection',)
    parser.add_argument('-t', '--training', action='store_true')
    training = vars(parser.parse_args())['training']

    if training:
        config = {
            "hidden": tune.choice([256, 512]),
            "lr": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([50, 64]),
            "epochs" : tune.choice([20])
        }

        best_trial = model_selection(config)

        retrain(best_trial.config) # retrain and save model params
        model = MLPNet(hidden = best_trial.config["hidden"])
    else:
        model = MLPNet(hidden = 512)

    model = load_model(model)
        
    test_acc = test_model(model, dir_path=DATASET_PATH)
    print("Test set accuracy: {}".format(test_acc))

if __name__ == "__main__":
    main()