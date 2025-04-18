import os
import random
import ray.train
import torch
import torchmetrics
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose

import ray
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, kernel_size=3)
        self.fc = torch.nn.Linear(192, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return torch.log_softmax(x, dim=1)

class MLPNet(torch.nn.Module):
    def __init__(self, input_size = 28, hidden = 512, output_size = 10, seed = None):
        super().__init__()
        
        gen = torch.Generator()
        if seed is not None:
            gen = gen.manual_seed(seed)

        self.input_size = input_size
        self.l1 = torch.nn.Linear(input_size * input_size, hidden)
        self.l2 = torch.nn.Linear(hidden, output_size)

        torch.nn.init.kaiming_uniform_(self.l1.weight, generator = gen)
        torch.nn.init.kaiming_uniform_(self.l2.weight, generator = gen)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, self.input_size * self.input_size)
        x = torch.relu(self.l1(x))
        x = torch.softmax(self.l2(x), 1)
        return x

def print_metrics(loss: torch.Tensor, accuracy: torch.Tensor, epoch: int) -> None:
    metrics = {"loss": loss.item(), "accuracy": accuracy.item(), "epoch": epoch}
    if ray.train.get_context().get_world_rank() == 0:
        print(metrics)
    return metrics

def save_checkpoint_and_metrics(model: torch.nn.Module, metrics: dict[str, float]) -> None:
    checkpoint = None
    if ray.train.get_context().get_world_rank() == 0:
        torch.save(
            model.module.state_dict(),
            os.path.join(os.path.abspath("/home/marco/Documenti/University/SDC/project/SDC-project/models"), "model.pt")
        )
        checkpoint = ray.train.Checkpoint.from_directory(os.path.abspath("/home/local/ADUNIPI/y.andriaccio/gitDatabase/marco-scalable/models"))

    ray.train.report(
        metrics,
        checkpoint=checkpoint,
    )

def run_training(config: dict):
    criterion = CrossEntropyLoss()
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    model = load_model(config['seed'])
    optimizer = Adam(model.parameters(), lr=1e-5)
    
    global_batch_size = config["global_batch_size"]
    batch_size = global_batch_size // ray.train.get_context().get_world_size()
    data_loader = build_data_loader_train(batch_size= batch_size)
    
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(model.device)

    for epoch in range(config["num_epochs"]):
        data_loader.sampler.set_epoch(epoch)

        for images, labels in data_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc(outputs, labels)

        accuracy = acc.compute()

        metrics = print_metrics(loss, accuracy, epoch)
        acc.reset()

    save_checkpoint_and_metrics(model, metrics)

def run_test(model: torch.nn.Module):
    correct, total = 0, 0

    # global_batch_size = config["global_batch_size"]
    # batch_size = global_batch_size // ray.train.get_context().get_world_size()
    data_loader = build_data_loader_test()

    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def load_model(seed: int) -> torch.nn.Module:
    # model = ConvNet()
    model = MLPNet(seed= seed)
    model = ray.train.torch.prepare_model(model)
    return model

def build_data_loader_train(batch_size: int) -> DataLoader:
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    data = MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    loader = ray.train.torch.prepare_data_loader(loader)
    return loader

def build_data_loader_test() -> DataLoader:
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    data = MNIST(root="./data", train=False, download=True, transform=transform)
    loader = DataLoader(data, shuffle=True, drop_last=True)

    # loader = ray.train.torch.prepare_data_loader(loader)
    return loader

def main():
    import os

    scaling_config = ScalingConfig(
        num_workers = 2,
        use_gpu = False
    )
    run_config = RunConfig(
        storage_path = os.path.abspath("./artifacts"),
        name = "distributed-mnist"
    )
    train_loop_config = {
        "seed": 42,
        "num_epochs": 20,
        "global_batch_size": 50
    }

    trainer = TorchTrainer(
        run_training,
        scaling_config = scaling_config,
        run_config = run_config,
        train_loop_config = train_loop_config,
    )
    result = trainer.fit()
    print(result.metrics_dataframe)
    print(result.checkpoint)

    model = MLPNet()
    with result.checkpoint.as_directory() as dir:
        model.load_state_dict( torch.load(os.path.join(dir, 'model.pt'), weights_only= True) )

    result = run_test(model)
    print("Accuracy: ", result)

if __name__ == "__main__":
    main()