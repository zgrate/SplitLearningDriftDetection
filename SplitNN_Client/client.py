from datetime import datetime
from os import path
from time import sleep

import torch
from torch import nn, Tensor

from SplitNN_Client.data_provider import AbstractDataInputStream, MNISTDataInputStream, get_test_training_data
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
            if "weight" in module.__dir__():
                nn.init.normal_(module.weight)
            if "bias" in module.__dir__():
                nn.init.normal_(module.bias)

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

    def __init__(self, client_id, data_input_stream: AbstractDataInputStream, folder: str = "folder"):
        self.model = ClientModel()
        self.epoch = 0
        self.folder = folder
        self.client_id = client_id
        self.server = ServerConnection(client_id)
        self.server_url = "http://localhost:8000"
        self.data_input_stream = data_input_stream
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.error_counter = 0

    def reset_local_nn(self):
        self.model.reset_nn()
        self.epoch = 0

    def train_round(self, depth=False):
        self.model.train()
        data = self.data_input_stream.get_data_part()
        output: Tensor = self.model(data.train_data)
        if output.isnan().any() or output.isinf().any():
            self.log("NaN/Inf values detected. Reseting local NN")
            # TODO: Send Reset to Server
            if depth:
                raise Exception("NaN/Inf values detected and tried to fix. I should instruct server of a problem")

            return self.train_round(depth=True)

        # print(data, output)
        response = self.server.train_request(output, data.train_labels, self.epoch)

        if response['gradients'] == []:
            self.log("OK we have a fuckup. Lets wait a bit for other clients")
            self.error_counter += 1
            if self.error_counter >= 5:
                self.log("Resseting NN")
                self.reset_local_nn()
                self.server.report_reset_nn()
                self.error_counter = 0

            sleep(3)

        else:
            server_gradients = torch.tensor(response['gradients'])
            output.backward(server_gradients)
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.epoch += 1

            self.log("GOT LOSS", response['loss'])

    def test_nn(self):
        self.model.train()
        data = self.data_input_stream.get_data_part()
        output = self.model(data.test_data)
        if output.isnan().any() or output.isinf().any():
            self.log("NaN/Inf values detected. Local NN should be reset")
            return False

        response = self.server.test_request(output, data.test_labels)
        self.log("TEST_LOSS", response['loss'])
        return response['loss']

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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log("Close file at", datetime.now())
        self.logger_file.close()


def run_client(client_id, max_client, folder):
    mnist_input = MNISTDataInputStream(*get_test_training_data(client_id, max_client))
    d = mnist_input.get_data_part()

    with TrainingSuite(client_id, mnist_input, folder) as t:
        while True:
            try:
                t.train_round()
                # t.test_nn()
                # t.log("PREDICTED LABEL: ", t.predict(d.train_data[0].view(1, -1)), "EXPECTED", d.test_labels[0])
                sleep(1)
            except KeyboardInterrupt:
                break
