# Cut model
# Initial Training on Client Side: Each client trains its portion of the model (up to the cut layer) using its local data. This step generates what is referred to as "smashed data," which represents intermediate outputs of the mode
# Server Model Training
# Client Model Update
import datetime
from typing import Mapping, Any
import copy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from torchvision.transforms import transforms
from torchvision.transforms.v2 import ToTensor
from torchvision.datasets import MNIST
import sys

from dataloader import VerticalDataLoader
from neural_network.nn.my.data_logger import ServerDataLogger, ClientDataLogger
from split_data import add_ids


# Client -> nn.Sequential 1. model
# Server -> nn.Sequential 2. modlel

# SErver -> random params on 2. model
# Server -> state_dict -> client
# Client -> nn.Sequential (1. model, 2. model)
# Clinet -> evaluate like NN
# Client -> 2.model.state_dict -> Server

# Server






base_server_model = nn.Sequential(nn.Dropout(0.25), nn.Flatten(), nn.Linear(9216, 128), nn.ReLU(), nn.Linear(128, 10), nn.LogSoftmax(dim=1))

class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.model = nn.Sequential()

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True, assign: bool = False):
        return self.model.load_state_dict(state_dict, strict=strict, assign=assign)
    
    def forward(self, x):
        return self.model(x)


class ClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.25),
            # nn.Flatten(1),
            # nn.Linear(9216, 128),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(128, 10),
            # nn.LogSoftmax(1)
        )
        # self.model = nn.Sequential(
        #     # ClientLayer(),
        #     nn.Flatten(),
        #     nn.Linear(784, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 600),
        #     nn.ReLU(),
        #     # nn.Linear(CLIENT_OUTPUT_INPUTS, 10)
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Client():
    def __init__(self,id, client_file="client_data.csv"):
        self.model = ClientModel()
        self.id = id
        self.data = None
        self.test = None
        self.logger = ClientDataLogger(id, f"{self.id}+{client_file}")
        self.logger.__enter__()
        self.logger.log_client_message("Starting client...")


    def init_dataset(self, *args, **kwargs):
        self.logger.log_client_message("Loading data...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        mnist = MNIST("mnists/", download=True, transform=transform, train=True)
        mnist_test = MNIST("mnists/", download=True, transform=transform, train=False)

        self.data = DataLoader(Init.filter_by_target(mnist, args[0]))
        self.test = DataLoader(mnist_test)

        self.logger.log_client_message("Data loading complete!")

    def __exit__(self):
        self.logger.__exit__()
    def fit(self):
        embedding = self.model(self.data)
        return embedding.detach().numpy()

    def evaluate(self, state_dict):
        self.logger.log_client_transfer(sys.getsizeof(state_dict))
        server_model = nn.Sequential(nn.Dropout(0.25), nn.Flatten(), nn.Linear(9216, 128), nn.ReLU(), nn.Linear(128, 10), nn.LogSoftmax(dim=1))
        server_model.load_state_dict(state_dict)
        full_model = nn.Sequential(self.model, server_model)

        optimizer = torch.optim.SGD(full_model.parameters(), lr=0.01)
        cel = torch.nn.CrossEntropyLoss()

        self.logger.log_client_message("Starting training...")

        for batch_idx, (data, target) in enumerate(self.data):
            if batch_idx % 100 == 0:
                self.logger.log_client_message(f"Batch {batch_idx}")
            data, target = data.to(torch.device("cpu")), target.to(torch.device("cpu"))
            optimizer.zero_grad()
            output = full_model(data)
            loss = cel(output, target)
            loss.backward()
            optimizer.step()
            # print(self.id, loss)

        self.logger.log_client_message("STarting testing...")

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test:
                data, target = data.to(torch.device("cpu")), target.to(torch.device("cpu"))
                output = full_model(data)
                test_loss += cel(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(self.test.dataset)

        self.logger.log_client_message("Full run performed, going back to server")

        # print(full_model[1].state_dict())

        self.logger.log_client_parameters(test_loss, correct, len(self.test.dataset))

        return test_loss, correct, len(self.test.dataset), full_model[1].state_dict()


class Init:

    def __init__(self, clientsNum):
        self.clientsNum = clientsNum

    @classmethod
    def filter_by_target(cls, dataset, target_classes):
        indices = [i for i, (_, target) in enumerate(dataset) if target in target_classes]
        return Subset(dataset, indices)


    def init_dataset(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        mnist = MNIST("mnists/", download=True, transform=transform, train=True)
        mnist_test = MNIST("mnists/", download=True, transform=transform, train=False)

        mnsts = [self.filter_by_target(mnist, [0, 1, 2, 3]), self.filter_by_target(mnist, [4,5,6]), self.filter_by_target(mnist, [7, 8,9])]


    def start(self):
        print("Initializing workers...")


        # le = len(mnist.data) / self.clientsNum
        clients = []


        mnsts = [[0, 1, 2, 3],[4,5,6],[7, 8,9]]

        with ServerDataLogger() as logger:


            for i in range(self.clientsNum):
                # start = int(i * le)
                # end = int((i + 1) * le)

                # data_loader = DataLoader(TensorDataset(mnist.data[start:end].float(), mnist.targets[start:end]))
                logger.log_server_message(f"Starting Client {i}")
                client = Client(i)
                client.init_dataset(mnsts[i])
                clients.append(client)

            server = ServerModel()

            logger.log_server_message(f"Starting NN with {self.clientsNum}")

            run = 0

            while True:
                run += 1
                logger.log_server_start_next_epoch(run)
                duration = datetime.datetime.now()

                data = base_server_model.state_dict()
                for client in clients:
                    test, loss, lengh, server_data = client.evaluate(data)
                    logger.log_server_transfer(sys.getsizeof(server_data))
                    data = server_data

                logger.log_epoch_duration(run, (datetime.datetime.now() - duration).total_seconds())







class StrategyWorker():
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()
        self.server_model = ServerModel()


class Worker:
    def __init__(self):
        self.model = None
        self.data = None

    def load_local_data(self, data):
        self.data = data
        pass

    def receive_model(self, model):
        self.model = model

    def local_train(self):
        pass


class Server:

    def __init__(self, server_model):
        self.server_model = server_model


init = Init(3)
init.start()

exit()

import pandas as pd
from torchvision.datasets import MNIST

dataset_mnist = add_ids(MNIST)("mnists/", download=True, transform=ToTensor())

train_data = VerticalDataLoader(dataset_mnist, batch_size=10)

print(next(iter(train_data)))

w = Worker()
w.load_local_data(train_data)

torch.manual_seed(0)

# Define our model segments

input_size = 784
hidden_sizes = [128, 640]
output_size = 10

models = [
    nn.Sequential(
        # ClientLayer(),
        nn.Flatten(),
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
    ),
    nn.Sequential(nn.Linear(hidden_sizes[1], output_size), nn.LogSoftmax(dim=1)),
]

# Create optimisers for each segment and link to them
optimizers = [
    optim.SGD(model.parameters(), lr=0.03, )
    for model in models
]

w.receive_model(models[0])

# train
pred = models[0](next(iter(train_data))[0][0])

print()
