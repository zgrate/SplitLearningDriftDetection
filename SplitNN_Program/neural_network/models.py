import json
import os

import torch
from django.db import models
from torch import nn, Tensor, optim

from data_logger.models import TrainingLog
from drift_detection.drift_detectors import DriftDetectionSuite, SimpleAverageDriftDetection


# Create your models here.


class ServerModelWrapper(nn.Module):
    def __init__(self):
        super(ServerModelWrapper, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, out_features=128),
            nn.ReLU(),
            nn.Linear(128, out_features=100),
            nn.ReLU(),
            nn.Linear(100, out_features=10),
            # nn.LogSoftmax(dim=1)
        )

    def reset_nn(self):
        def init_normal(module):
            if "weight" in module.__dir__():
                nn.init.normal_(module.weight)
            if "bias" in module.__dir__():
                nn.init.normal_(module.bias)

        self.model.apply(init_normal)

    def forward(self, x):
        return self.model(x)


class ServerModel:

    def __init__(self, input_dict=None, drift_detection_suite=DriftDetectionSuite(SimpleAverageDriftDetection(filter_mode="test", client_only_mode=True))):
        self.model = ServerModelWrapper()
        self.drift_detection_suite = drift_detection_suite

        self.epoch = 0
        if input_dict is not None:
            print("Loading State DICT")
            self.model.load_state_dict(input_dict)
        else:
            self.model.reset_nn()

        self.optimizer = None
        self.reinit_optimiser(lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.error_counter = 0
        self.options = None

    def load(self, file):
        t = torch.load(file, weights_only=True)
        self.model.load_state_dict(t)
        # return ServerModel(input_dict=t)

    def save(self, target_file):
        os.makedirs(target_file, exist_ok=True)
        torch.save(global_server_model.model.state_dict(), os.path.join(target_file, "server.pt"))

    def reset_local_nn(self):
        TrainingLog(mode="reset", server_epoch=self.epoch).save()
        self.epoch = 0
        self.model.reset_nn()

    def optimizer_pass(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def reinit_optimiser(self, **options):
        print(options)
        self.optimizer = optim.SGD(self.model.parameters(), **options)
        print(self.optimizer)

    def train_input(self, input_list: list, input_labels: list, depth=0):
        self.model.train()
        input_data = torch.tensor(input_list, requires_grad=True)
        labels = torch.tensor(input_labels).long()

        output = self.model(input_data)

        loss = self.criterion(output, labels)
        loss.backward()

        gradients = input_data.grad.detach()
        return_empty = False
        if gradients.isnan().any() or gradients.isinf().any():
            print("Reset of NN needed. Reseting system....  ")
            self.error_counter += 1
            if self.error_counter >= 10:
                print("OK now we should definitly reset NN. Its bad!")
                self.reset_local_nn()
                self.error_counter = 0

            if depth == 0:
                print("For now we will skip one or two iterations...")
                return_empty = True
            # elif depth == 2:
            #     raise Exception("Repeated error! Depth loop prevention!")
            # else:
            #     self.reset_local_nn()
            #
            #     return self.train_input(input_list, input_labels, depth=depth+1)

        if return_empty:
            gradients = []
            loss = 0
        else:
            gradients = gradients.numpy()

        self.optimizer_pass()

        self.epoch += 1
        return gradients, float(loss)

    def test(self, input_list: list, input_labels: list):
        self.model.eval()
        input_data = torch.tensor(input_list)
        labels = torch.tensor(input_labels).long()
        output = self.model(input_data)
        loss = self.criterion(output, labels)

        if loss.isnan().any() or loss.isinf().any():
            print("Reset of NN Needed")
            return None

        return float(loss)

    def predict(self, intput_list: list):
        input_data = torch.tensor(intput_list)
        self.model.eval()
        output = self.model(input_data)
        if output.isnan().any() or output.isinf().any():
            print("Predition is off board! Need fixing")
            return None

        return output



global_server_model = ServerModel()
