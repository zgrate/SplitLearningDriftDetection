import dataclasses
import os
from datetime import datetime

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from RunnersTesting.shared.data_provider import get_test_training_data, MNISTDataInputStream, DataInputDataset


class ServerModelWrapper(nn.Module):
    def __init__(self):
        super(ServerModelWrapper, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, out_features=128),
            nn.ReLU(),
            nn.Linear(128, out_features=100),
            nn.ReLU(),
            nn.Linear(100, out_features=10),
            nn.LogSoftmax(dim=1)
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

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


@dataclasses.dataclass
class TrainingLog:
    created_at: str

    loss: float
    epoch: int

    training_time: float

    def to_csv(self):
        dictionary = self.__dict__
        return ";".join([str(value) for key, value in dictionary.items()])


class CentralNNTraining:

    def log_training(self, log: TrainingLog):
        if self.training_stream is None or self.training_stream.closed:
            print("TRAINING STREAM DEAD")
        else:
            self.training_stream.write(f"{log.to_csv()}\n")
            print(log.to_csv())
            self.training_stream.flush()

    pass

    def log(self, *log):
        if self.logger_stream is None or self.logger_stream.closed:
            print("LOGGER FILE NOT OPENED")
        else:
            current_time = datetime.now()
            s = f"{current_time}:{' '.join([str(x) for x in log])}\n"
            print(s)
            self.logger_stream.write(s)
            self.logger_stream.flush()

    def __enter__(self, *args, **kwargs):
        self.logger_stream = open(self.logger_file, "w")
        self.training_stream = open(self.training_file, "w")
        self.training_stream.write(
            ";".join([key for key, value in TrainingLog(None, None, None, None).__dict__.items()]))
        self.training_stream.write("\n")
        return self

    def __exit__(self, *args, **kwargs):
        self.training_stream.close()
        self.logger_stream.close()

    def __init__(self, parent_folder=""):

        # Main Model Stuff
        self.model = nn.Sequential(
            ClientModel(),
            ServerModelWrapper()
        )
        self.optimiser = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()

        # Data
        self.input_data_stream = MNISTDataInputStream(*get_test_training_data(0, 1))
        self.data_loader = DataLoader(dataset=DataInputDataset(self.input_data_stream), batch_size=100)

        # Other Init Stuff for data logging
        self.epoch = 0
        self.folder = os.path.join(parent_folder, datetime.now().strftime("%Y%m%d_%H%M%s"))
        os.makedirs(self.folder, exist_ok=True)
        self.logger_file = os.path.join(self.folder, "logs.txt")
        self.training_file = os.path.join(self.folder, "training_logs.csv")

    def train_epoch(self):
        self.log("Start epoch", self.epoch)

        time_start = datetime.now()
        self.model.train()

        losses = []
        for batch, (X, y) in enumerate(self.data_loader):
            output = self.model(X)

            loss = self.loss_fn(output, y)
            loss.backward()

            self.optimiser.step()
            self.optimiser.zero_grad()

            losses.append(float(loss))

        time_stop = datetime.now()

        log = TrainingLog(created_at=str(datetime.now()), loss=float(sum(losses)/len(losses)), epoch=self.epoch,
                          training_time=(time_stop - time_start).total_seconds())

        self.log_training(log)
        self.epoch += 1


if __name__ == "__main__":
    with CentralNNTraining() as cc:
        print(cc.model)
        for i in range(300):
            cc.train_epoch()
