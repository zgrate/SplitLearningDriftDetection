import json
import os.path
import random
from datetime import datetime
from functools import partial
from os import path
from time import sleep

import torch
from torch import nn, Tensor
from torch.utils.data import Subset

from SplitNN_Client.data_provider import AbstractDataInputStream, MNISTDataInputStream, get_test_training_data, \
    DataInputDataset, DriftDatasetLoader, get_separated_by_labels
from SplitNN_Client.drifted_data_creator import add_noise, temporal_drift
from SplitNN_Client.drifting_simulation import RandomDrifter, AbstractDrifter
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

    def __init__(self, client_id, data_input_stream: DataInputDataset, optimizer=torch.optim.SGD,
                 learning_rate=0.001, folder: str = "folder", load_save_data_directory: str = None, load_only=False, reset_nn=True,
                 drifter_options=None):
        self.validation_data_loader = None
        self.training_data_loader = None
        self.validation_data = None
        self.training_data = None
        self.dataset = None
        if drifter_options is None:
            drifter_options = dict()
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
            self.swap_datasets(data_input_stream, drifter_options)
            # self.dataset = DataInputDataset(data_input_stream)
            # self.training_data, self.validation_data = torch.utils.data.random_split(self.dataset, [0.9, 0.1])
            # self.training_data_loader = DriftDatasetLoader(self.training_data, drifter=RandomDrifter(**drifter_options),
            #                                                         batch_size=len(self.training_data.indices))
            # self.validation_data_loader = DriftDatasetLoader(self.validation_data, drifter=RandomDrifter(**drifter_options),
            #                                                           batch_size=len(self.validation_data.indices))

        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.error_counter = 0
        self.last_comm_time = 0
        self.last_whole_training_time = 0

    def swap_datasets(self, dataset, drifter_function):
        if drifter_function is None:
            drifter_options = lambda x,y,_ : (x,y )

        self.dataset = dataset
        self.training_data, self.validation_data = torch.utils.data.random_split(self.dataset, [0.9, 0.1])
        self.training_data_loader = DriftDatasetLoader(self.training_data, drifter=AbstractDrifter(**drifter_options),
                                                       batch_size=len(self.training_data.indices))
        self.validation_data_loader = DriftDatasetLoader(self.validation_data, drifter=AbstractDrifter(**drifter_options),
                                                         batch_size=len(self.validation_data.indices))
        # self.validation_data = validation_data
        # self.validation_data_loader = DriftDatasetLoader(self.validation_data, drifter=AbstractDrifter(), batch_size=len(self.validation_data.indices))

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
        self.log("GOT TRAIN LOSS", loss)
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


def get_random_prediction_mnist(input_data: DataInputDataset):
    data, label = input_data.random_data_label()
    return data.view(1, 28, 28), label

def run_client(client_id, thread_runner):
    if thread_runner.all_props_dict['labels_filter'] is not None:
        mnist_input = DataInputDataset(MNISTDataInputStream(*get_separated_by_labels(thread_runner.all_props_dict['labels_filter'][client_id])))
    else:
        mnist_input = DataInputDataset(MNISTDataInputStream(*get_test_training_data(client_id, thread_runner.clients)))

    if thread_runner.all_props_dict['drift_type'] == "add_noise":
        drifted_data = DataInputDataset(MNISTDataInputStream(*get_test_training_data(client_id, thread_runner.clients)), drift_transformation=partial(add_noise, **thread_runner.all_props_dict['drifter_options']))

    elif thread_runner.all_props_dict['drift_type'] == "swap_domain":
        domain_id = client_id + 1
        if len(thread_runner.all_props_dict['labels_filter']) >= domain_id:
            domain_id = 0
        drifted_data = DataInputDataset(MNISTDataInputStream(*get_separated_by_labels(thread_runner.all_props_dict['labels_filter'][domain_id])))
    elif thread_runner.all_props_dict['drift_type'] == "temporal_drift":
        drifted_data = DataInputDataset(MNISTDataInputStream(*get_test_training_data(client_id, thread_runner.clients)), drift_transformation=partial(temporal_drift, **thread_runner.all_props_dict['drifter_options']))

    # exit(0)mnist_input
    # print(mnist_input.data.train_data.shape)
    with TrainingSuite(client_id, mnist_input, learning_rate=thread_runner.client_learning_rate,
                       folder=thread_runner.folder, load_save_data_directory=thread_runner.all_props_dict['client_load_directory'], load_only=thread_runner.all_props_dict['load_only'], reset_nn=thread_runner.all_props_dict['reset_nn'], drifter_options=thread_runner.all_props_dict['drifter_options']) as t:
        list_of_last_results = []
        mode = "train"
        last_test_check = 0
        prediction_epoch = 0
        current_dataset = mnist_input

        if thread_runner.all_props_dict['start_drifting']:
            t.validation_data_loader.drift_active = True
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

                    # r = random.randint(0, len(mnist_input.data.train_data))
                    image, target_label = get_random_prediction_mnist(mnist_input)
                    p = t.predict(image)
                    pred = p['predicted'][0]
                    min_index, min_value = max(enumerate(pred), key=lambda x: x[1])
                    # print(pred)
                    if int(p['item']) == target_label:
                        print("Match", target_label)
                    else:
                        print("Prediction dont match:",  min_index, min_value, "label", target_label, p['item'])
                    sleep(5)

                if thread_runner.all_props_dict['mode'] == "normal_runner":
                    if prediction_epoch == thread_runner.all_props_dict['predict_epochs_swap']:
                        if len(thread_runner.all_props_dict['drifting_clients']) == 0 or client_id in thread_runner.all_props_dict['drifting_clients']:
                            t.log("Start the DRIFT!")
                            t.swap_datasets(drifted_data)
                            current_dataset = drifted_data
                            prediction_epoch += 1
                            sleep(5)

                    if mode == "train":
                        t.log("Training round")
                        loss = t.train_round(False)
                        if loss <= thread_runner.all_props_dict['target_loss']:
                            t.log("We are trained! Going back to predicting...")
                            mode = "predict"

                    elif mode == "predict":
                        image, target_label = get_random_prediction_mnist(current_dataset)
                        p = t.predict(image)
                        if thread_runner.all_props_dict['check_mode'] == "prediction":
                            list_of_last_results.insert(0, (1 if int(p['item']) == target_label else 0))

                            failures, all_tests = thread_runner.all_props_dict['prediction_errors_count']
                            list_of_last_results = list_of_last_results[:all_tests]
                            if client_id == 0:
                                print(list_of_last_results, all_tests-sum(list_of_last_results), failures)
                            if len(list_of_last_results) == all_tests and all_tests - sum(list_of_last_results) >= failures:
                                t.log("We have a failures! Double check with tests!")
                                loss = t.test_nn()
                                if loss > thread_runner.all_props_dict['target_loss']:
                                    t.log("Target loss not met ", loss, ". Training!")
                                    mode = "train"
                                else:
                                    t.log("is OK")

                                list_of_last_results = []

                        t.log("Prediction ", p['item'], "should be ", target_label)
                        prediction_epoch += 1
                        sleep(1)
                        if prediction_epoch % 10 == 0:
                            t.log("Sanity test of NN", t.test_nn())

                        if thread_runner.all_props_dict['check_mode'] == "testing":
                            last_test_check += 1
                            if  last_test_check >= thread_runner.all_props_dict['check_testing_repeating']:
                                t.log("Testing the predicted NN")
                                last_test_check = 0
                                loss = t.test_nn()
                                if loss > thread_runner.all_props_dict['target_loss']:
                                    t.log("Target loss not met ", loss, ". Training!")
                                    mode = "train"


            except KeyboardInterrupt:
                break
        t.log("Stopping client")
