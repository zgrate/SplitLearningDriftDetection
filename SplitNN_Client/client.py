from time import sleep

import torch
from torch import nn

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
            def init_normal(module):
                if "weight" in module.__dir__():
                    print("Weight")
                    nn.init.normal_(module.weight)
                if "bias" in module.__dir__():
                    print("bias")
                    nn.init.normal_(module.bias)


            self.model.apply(init_normal)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Constants:
    API_TRAIN = "/train"


server = ServerConnection("hahahahah")


class TrainingSuite:

    def __init__(self, data_input_stream: AbstractDataInputStream):
        self.model = ClientModel()
        self.server_url = "http://localhost:8000"
        self.data_input_stream = data_input_stream
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def train_round(self):
        self.model.train()
        data = self.data_input_stream.get_data_part()
        output = self.model(data.train_data)
        print(data, output)
        response = server.train_request(output, data.train_labels)


        server_gradients = torch.tensor(response['gradients'])

        output.backward(server_gradients)
        self.optimizer.step()


        print("GOT LOSS", response['loss'])

    def test_nn(self):
        self.model.train()
        data = self.data_input_stream.get_data_part()
        output = self.model(data.test_data)
        response = server.test_request(output, data.test_labels)
        print("TEST_LOSS", response['loss'])

    def predict(self, input):
        self.model.eval()
        output = self.model(input)
        return server.predict_request(output)


client_id = int(input("Client ID "))
t = TrainingSuite(MNISTDataInputStream(*get_test_training_data(client_id, 5)))
d = t.data_input_stream.get_data_part()
while(True):

    t.train_round()
    t.test_nn()
    print("PREDICTED LABEL: ", t.predict(d.train_data[0].view(1, -1)), "EXPECTED", d.test_labels[0])
    sleep(1)


def train_run():
    pass
