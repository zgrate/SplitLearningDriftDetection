#Cut model
#Initial Training on Client Side: Each client trains its portion of the model (up to the cut layer) using its local data. This step generates what is referred to as "smashed data," which represents intermediate outputs of the mode
#Server Model Training
#Client Model Update


import torch
from torch import nn, optim
from torchvision.transforms.v2 import ToTensor

from dataloader import VerticalDataLoader
from split_data import add_ids


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
    optim.SGD(model.parameters(), lr=0.03,)
    for model in models
]

w.receive_model(models[0])


#train
pred = models[0](next(iter(train_data))[0][0])

print()

