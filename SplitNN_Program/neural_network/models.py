import json

import torch
from django.db import models
from torch import nn, Tensor, optim


# Create your models here.


class ServerModelWrapper(nn.Module):
    def __init__(self):
        super(ServerModelWrapper, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, out_features=50),
            nn.ReLU(),
            nn.Linear(50, out_features=10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


class ServerModel:

    def __init__(self, input_dict=None):
        self.model = ServerModelWrapper()
        if input_dict is not None:
            self.model.load_state_dict(input_dict)

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            input_dict = json.load(f)
            return ServerModel(input_dict)

    def save(self, target_file):
        with open(target_file, 'w') as f:
            json.dump(self.model.state_dict(), f)

    def train_input(self, input_list: list, input_labels: list):
        self.model.train()
        input_data = torch.tensor(input_list, requires_grad=True)
        labels = torch.tensor(input_labels).long()

        self.optimizer.zero_grad()

        output = self.model(input_data)
        print(output)
        loss = self.criterion(output, labels)
        loss.backward()
        self.optimizer.step()

        gradients = input_data.grad

        return gradients, float(loss)

    def test(self, input_list: list, input_labels: list):
        self.model.train()
        input_data = torch.tensor(input_list)
        labels = torch.tensor(input_labels).long()
        output = self.model(input_data)
        loss = self.criterion(output, labels)
        return float(loss)

    def predict(self, intput_list: list):
        input_data = torch.tensor(intput_list)
        self.model.eval()
        return self.model(input_data)


global_server_model = ServerModel()
