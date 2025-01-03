import json
import os.path
import random
from datetime import datetime
from os import path
from time import sleep

import torch
from torch import nn, Tensor
from torch.utils.data import Subset

from SplitNN_Client.data_provider import AbstractDataInputStream, MNISTDataInputStream, get_test_training_data, \
    DataInputDataset
from SplitNN_Client.server_connection import ServerConnection


class ClientModel(nn.Module):
    def __init__(self, file=None):
        super().__init__()
        self.model = nn.Sequential(
            # ClientLayer(),
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 100),
            nn.ReLU(),
        )
        if file is None:
            self.reset_nn()

    def reset_nn(self):

        def init_normal(module):
            if "reset_parameters" in module.__dir__():
                module.reset_parameters()

        self.model.apply(init_normal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Constants:
    API_TRAIN = "/train"


class TrainingSuite:

    def log(self, *log):
        if self.logger_file is None or self.logger_file.closed:
            print("LOGGER FILE NOT OPENED")
        else:
            current_time = datetime.now()
            s = f"{current_time}:{self.client_id}:{' '.join([str(x) for x in log])}\n"
            print(s)
            self.logger_file.write(s)
            self.logger_file.flush()

    def __init__(self, client_id, data_input_stream: AbstractDataInputStream, optimizer=torch.optim.SGD,
                 learning_rate=0.001, folder: str = "folder", load_save_data_directory: str = None, load_only=False, reset_nn=True):
        self.model = ClientModel()
        self.epoch = 0
        self.folder = folder
        self.client_id = client_id
        self.server = ServerConnection(client_id)
        self.server_url = "http://localhost:8000"
        self.load_save_data_directory = load_save_data_directory
        self.load_only = load_only
        self.reset_nn = reset_nn

        if data_input_stream is not None:
            self.dataset = DataInputDataset(data_input_stream)
            self.training_data, self.validation_data = torch.utils.data.random_split(self.dataset, [0.9, 0.1])
            self.training_data_loader = torch.utils.data.DataLoader(self.training_data,
                                                                    batch_size=len(self.training_data.indices))
            self.validation_data_loader = torch.utils.data.DataLoader(self.validation_data,
                                                                      batch_size=len(self.validation_data.indices))

        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.error_counter = 0
        self.last_comm_time = 0
        self.last_whole_training_time = 0

    def reset_local_nn(self):
        self.model.reset_nn()
        self.epoch = 0

    def train_round(self, synchronizer, depth=False):
        if synchronizer:
            while self.server.current_client() != self.client_id:
                sleep(0.5)

        client_start_time = datetime.now()
        self.model.train()
        # X, y = self.training_data_loader[0]
        # print(X.shape, y.shape)
        losses = []
        for X, y in iter(self.training_data_loader):
            output: Tensor = self.model(X)

            if output.isnan().any() or output.isinf().any():
                self.log("NaN/Inf values detected. Reseting local NN")

                # TODO: Send Reset to Server

            # print(data, output)
            comms_start_time = datetime.now()
            response = self.server.train_request(output, y, self.epoch, self.last_comm_time,
                                                 self.last_whole_training_time)
            self.last_comm_time = (datetime.now() - comms_start_time).total_seconds()
            losses.append(response['loss'])
            server_gradients = torch.tensor(response['gradients'])
            output.backward(server_gradients)
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.epoch += 1
        loss = sum(losses) / len(losses)
        self.log("GOT TEST LOSS", loss)
        #TODO: I should report avrg loss, not loss of each iteration here!
        #Only the case when batching

        self.last_whole_training_time = (datetime.now() - client_start_time).total_seconds()
        return loss

    def test_nn(self):
        print("TESTING NN")
        self.model.eval()

        with torch.no_grad():
            losses = []
            for X, y in self.validation_data_loader:
                output = self.model(X)
                if output.isnan().any() or output.isinf().any():
                    self.log("NaN/Inf values detected. Local NN should be reset")
                    return False

                response = self.server.test_request(output, y, self.epoch)
                losses.append(response['loss'])

        self.log("TEST_LOSS", sum(losses) / len(losses))
        return sum(losses) / len(losses)

    def predict(self, input):
        self.model.eval()
        output = self.model(input)
        if output.isnan().any() or output.isinf().any():
            self.log("NaN/Inf values detected. Local NN should be reset")
            return []
        return self.server.predict_request(output)

    def __enter__(self):
        self.logger_file = open(path.join(self.folder, f"client_{self.client_id}.log"), "w")
        self.log("File opened at", datetime.now())
        # print(self.load_save_data_directory, self.load_only, self.reset_nn)
        if self.load_save_data_directory and not self.reset_nn:
            p = path.join(self.load_save_data_directory, f"client_{self.client_id}.pt")
            if os.path.exists(p):
                self.log("Loading file ", p, "created at", os.path.getctime(p))
                t = torch.load(p, weights_only=True)
                self.model.load_state_dict(t)
                self.model.eval()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log("Close file at", datetime.now())
        self.logger_file.close()
        torch.save(self.model.state_dict(), path.join(self.folder, f"client_state_{self.client_id}.pt"))

        if self.load_save_data_directory and not self.load_only:
            os.makedirs(self.load_save_data_directory, exist_ok=True)
            torch.save(self.model.state_dict(), path.join(self.load_save_data_directory, f"client_{self.client_id}.pt"))


def run_client(client_id, thread_runner):
    mnist_input = MNISTDataInputStream(*get_test_training_data(client_id, thread_runner.clients))
    # d = mnist_input.get_data_part()
    print(mnist_input.data.train_data.shape)
    with TrainingSuite(client_id, mnist_input, learning_rate=thread_runner.client_learning_rate,
                       folder=thread_runner.folder, load_save_data_directory=thread_runner.all_props_dict['client_load_directory'], load_only=thread_runner.all_props_dict['load_only'], reset_nn=thread_runner.all_props_dict['reset_nn']) as t:
        t.test_nn()
        while not thread_runner.get_global_stop():
            try:
                if thread_runner.all_props_dict['mode'] == "train":
                    loss = t.train_round(thread_runner.sync_mode)
                    thread_runner.client_response(client_id, {"loss": loss})

                if thread_runner.all_props_dict['mode'] == "test":
                    t.test_nn()
                    sleep(3)
                    # if t.test_nn() < 0.3:
                    #     r = random.randint(0, len(mnist_input.data.test_data))
                    #     print("Prediction", t.predict(mnist_input.data.test_data[r]), mnist_input.data.test_labels[r])

                if thread_runner.all_props_dict['mode'] == "predict_random":
                    r = random.randint(0, len(mnist_input.data.train_data))
                    target_label = mnist_input.data.train_labels[r]
                    p = t.predict(mnist_input.data.train_data[r].view(1, 28, 28))
                    pred = p['predicted'][0]
                    min_index, min_value = max(enumerate(pred), key=lambda x: x[1])
                    # print(pred)
                    if int(p['item']) == target_label:
                        print("Match", target_label)
                    else:
                        print("Prediction dont match:",  min_index, min_value, "label", mnist_input.data.train_labels[r], p['item'])
                    sleep(5)

            except KeyboardInterrupt:
                break
        t.log("Stopping client")
