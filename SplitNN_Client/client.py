import json
import os.path
import random
from datetime import datetime
from functools import partial
from os import path
from time import sleep

import torch
from sympy.codegen import Print
from torch import nn, Tensor
from torch.utils.data import Subset

from SplitNN_Client.data_provider import get_test_training_data
from SplitNN_Client.nn_models import CIDARRecommendedNetwork1
from data_provider import AbstractDataInputStream, DividedDataInputStream, \
    DataInputDataset, DriftDatasetLoader, get_mnist_separated_by_labels
from drifted_data_creator import add_noise, temporal_drift
from drifting_simulation import RandomDrifter, AbstractDrifter
from nn_models import ClientServerModel0, ClientServerModel1, ClientServerModel2, ClientServerModel3, CNNClientServerModel1, CNNClientServerModel2
from server_connection import ServerConnection
from torchvision.datasets import MNIST, CIFAR10


models = {
    0: ClientServerModel0,
    1: ClientServerModel1,
    2: ClientServerModel2,
    3: ClientServerModel3,
    4: CNNClientServerModel1,
    5: CNNClientServerModel2,
    6: CIDARRecommendedNetwork1
}


target_dataset_details = {
    "mnist": {
        "dataset_input": partial(MNIST, "./mnist", download=True),
        "image_size": (28,28),
        "image_resolution": 1
    },
    "cifar10": {
        "dataset_input": partial(CIFAR10, "./cifar10", download=True),
        "image_size": (32,32),
        "image_resolution": 3
    }
}


class ClientModel(nn.Module):
    def __init__(self, file=None, model_type:int=1):
        super().__init__()
        self.model = models[model_type].client()
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

optimisers = {
    "sgd": torch.optim.SGD,
    "adamw": torch.optim.AdamW
}


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

    def __init__(self, client_id, optimiser="sgd", server_opt_options=dict(),
                 learning_rate=0.001, folder: str = "folder", load_save_data_directory: str = None, load_only=False, reset_nn=True, selected_model=1, runner_settings=None):
        self.selected_model = selected_model
        self.validation_data_loader = None
        self.training_data_loader = None
        self.validation_data = None
        self.training_data = None
        self.dataset = None
        self.model = ClientModel(model_type=selected_model)
        self.epoch = 0
        self.folder = folder
        self.client_id = client_id
        self.server = ServerConnection(client_id)
        self.server_url = "http://localhost:8000"
        self.load_save_data_directory = load_save_data_directory
        self.load_only = load_only
        self.reset_nn = reset_nn
        self.runner_settings = runner_settings

            # self.dataset = DataInputDataset(data_input_stream)
            # self.training_data, self.validation_data = torch.utils.data.random_split(self.dataset, [0.9, 0.1])
            # self.training_data_loader = DriftDatasetLoader(self.training_data, drifter=RandomDrifter(**drifter_options),
            #                                                         batch_size=len(self.training_data.indices))
            # self.validation_data_loader = DriftDatasetLoader(self.validation_data, drifter=RandomDrifter(**drifter_options),
            #                                                           batch_size=len(self.validation_data.indices))
        # optim = optimisers[optimiser]
        self.optimizer = models[selected_model].optimiser(self.model.parameters(), **models[selected_model].optimiser_parameters)

        self.error_counter = 0
        self.last_comm_time = 0
        self.last_whole_training_time = 0

    def set_dataset(self, dataset, drifter_function=None, start_epoch=0):
        if drifter_function is None:
            drifter_function = lambda x,y,_ : (x,y )

        self.dataset = dataset
        self.training_data, self.validation_data = torch.utils.data.random_split(self.dataset, [0.9, 0.1])
        self.training_data_loader = DriftDatasetLoader(self.training_data, drifter=drifter_function, start_epoch=start_epoch,
                                                       batch_size=len(self.training_data.indices) if models[self.selected_model].batch_size == 0 else models[self.selected_model].batch_size)
        self.validation_data_loader = DriftDatasetLoader(self.validation_data, drifter=drifter_function, start_epoch=start_epoch,
                                                         batch_size=len(self.validation_data.indices))
        # self.validation_data = validation_data
        # self.validation_data_loader = DriftDatasetLoader(self.validation_data, drifter=AbstractDrifter(), batch_size=len(self.validation_data.indices))

    def set_drifting(self, activate=True):
        self.training_data_loader.drift_active = activate
        self.validation_data_loader.drift_active = activate

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
        i = 0
        for X, y in iter(self.training_data_loader):
            self.log("Run ", i)

            i += 1
            output: Tensor = self.model(X.reshape(-1, target_dataset_details.get(self.runner_settings.dataset).get("image_resolution"), *target_dataset_details.get(self.runner_settings.dataset).get("image_size")))
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

    def test_nn(self, prediction_epoch=None, type="test"):
        print("TESTING NN")
        self.model.eval()
        any_drift_attempt = False
        with torch.no_grad():
            losses = []
            for X, y in self.validation_data_loader:
                output = self.model(X)
                if output.isnan().any() or output.isinf().any():
                    self.log("NaN/Inf values detected. Local NN should be reset")
                    return False

                response = self.server.test_request(output, y, self.epoch, type)

                if response['client_drifting']:
                    print("its drifting!")
                    any_drift_attempt = True

                losses.append(response['loss'])

        self.log("TEST_LOSS", sum(losses) / len(losses))
        return sum(losses) / len(losses), any_drift_attempt

    def predict(self, input, target_label: Tensor=None, prediction_epoch=None):
        if prediction_epoch is None:
            prediction_epoch = self.epoch

        self.model.eval()
        output = self.model(input)
        if output.isnan().any() or output.isinf().any():
            self.log("NaN/Inf values detected. Local NN should be reset")
            return []

        return self.server.predict_request(output, prediction_epoch, target_label.item())

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
            else:
                self.log("WARNING! FILE NOT FOUND", p)
                sleep(5)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log("Close file at", datetime.now())
        self.logger_file.close()
        torch.save(self.model.state_dict(), path.join(self.folder, f"client_state_{self.client_id}.pt"))

        if self.load_save_data_directory and not self.load_only:
            os.makedirs(self.load_save_data_directory, exist_ok=True)
            torch.save(self.model.state_dict(), path.join(self.load_save_data_directory, f"client_{self.client_id}.pt"))


def get_random_prediction(input_data: DataInputDataset, dataset_name, drifter_function=None, prediction_epoch=0, enable_drift=False):
    if drifter_function is None or not enable_drift:
        drifter_function = lambda x, y, _: (x, y)

    data, label = input_data.random_data_label()
    return drifter_function(data.view(1, *target_dataset_details[dataset_name].get("image_size")), label, prediction_epoch)

def get_random_prediction_cifar_10(input_data: DataInputDataset, drifter_function=None, prediction_epoch=0, enable_drift=False):
    if drifter_function is None or not enable_drift:
        drifter_function = lambda x, y, _: (x, y)

    data, label = input_data.random_data_label()
    return drifter_function(data.view(1, 32, 32), label, prediction_epoch)

def run_client(client_id, thread_runner):
    from runner import SplitLearningRunner
    thread_runner: SplitLearningRunner

    if thread_runner.runner_settings.labels_filter is not None:
        input_data = DataInputDataset(DividedDataInputStream(*get_mnist_separated_by_labels(thread_runner.runner_settings.labels_filter[client_id])))
    else:
        input_data = DataInputDataset(DividedDataInputStream(*get_test_training_data(client_id, thread_runner.runner_settings.clients, input_dataset=target_dataset_details[thread_runner.runner_settings.dataset].get("dataset_input")())))

    drifter_function = None
    drifted_data = None


    if thread_runner.runner_settings.drift_type == "add_noise":
        drifter_function = partial(add_noise, **thread_runner.runner_settings.drifter_options)
        # drifted_data = DataInputDataset(MNISTDataInputStream(*get_test_training_data(client_id, thread_runner.clients)), drift_transformation=drifter_function)

    elif thread_runner.runner_settings.drift_type == "half_clients_drift" and client_id % 2 == 0:
        drifter_function = partial(add_noise, **thread_runner.runner_settings.drifter_options)

    elif thread_runner.runner_settings.drift_type == "swap_domain":
        domain_id = client_id + 1
        if len(thread_runner.runner_settings.labels_filter) >= domain_id:
            domain_id = 0
        if thread_runner.runner_settings.dataset == "mnist":
            drifted_data = DataInputDataset(DividedDataInputStream(*get_mnist_separated_by_labels(thread_runner.runner_settings.labels_filter[domain_id])))
        else:
            raise Exception("Not implemented yet")

    elif thread_runner.runner_settings.drift_type == "temporal_drift":
        drifter_function = partial(temporal_drift, **thread_runner.runner_settings.drifter_options)
    elif thread_runner.runner_settings.drift_type == "temporal_drift_client":
        drifter_function = partial(temporal_drift, **{**thread_runner.runner_settings.drifter_options, "max_time_steps": thread_runner.runner_settings.drifter_options.get("max_time_steps", 999) * (client_id+1)})

        # drifted_data = DataInputDataset(MNISTDataInputStream(*get_test_training_data(client_id, thread_runner.clients)), drift_transformation=partial(temporal_drift, **thread_runner.runner_settings.drifter_options))
    else:
        drifter_function = lambda x, y, _: (x, y)

    # exit(0)input_data
    # print(input_data.data.train_data.shape)
    with TrainingSuite(client_id, optimiser=thread_runner.runner_settings.optimiser, server_opt_options= thread_runner.runner_settings.server_optimiser_options, learning_rate=thread_runner.client_learning_rate,
                       folder=thread_runner.folder, load_save_data_directory=thread_runner.runner_settings.client_load_directory, load_only=thread_runner.runner_settings.load_only, reset_nn=thread_runner.runner_settings.reset_nn, selected_model=thread_runner.runner_settings.selected_model, runner_settings=thread_runner.runner_settings) as t:
        t.set_dataset(input_data, drifter_function)
        list_of_last_results = []
        mode = "predict"
        last_test_check = 0
        prediction_epoch = 0
        current_dataset = input_data
        start_drift = False
        train_iterations_in_client = 0
        max_train_iterations = 100

        target_loss = thread_runner.runner_settings.target_loss

        if thread_runner.runner_settings.start_deviation_target != 0:
            target_loss, _ = t.test_nn()
            target_loss = target_loss + thread_runner.runner_settings.target_loss*target_loss


        # if thread_runner.runner_settings.start_drifting:
        #     t.validation_data_loader.drift_active = True
        while not thread_runner.get_global_stop():
            try:
                if thread_runner.runner_settings.mode == "train":
                    loss = t.train_round(thread_runner.sync_mode)
                    if thread_runner.runner_settings.absolute_target and loss < thread_runner.runner_settings.target_loss:
                        print("We are finished!")
                        break
                    else:
                        thread_runner.client_response(client_id, {"loss": loss})

                if thread_runner.runner_settings.mode == "test":
                    t.test_nn()
                    sleep(3)
                    break
                    # if t.test_nn() < 0.3:
                    #     r = random.randint(0, len(input_data.data.test_data))
                    #     print("Prediction", t.predict(input_data.data.test_data[r]), input_data.data.test_labels[r])

                    # r = random.randint(0, len(input_data.data.train_data))
                    if thread_runner.runner_settings.dataset == "mnist":
                        image, target_label = get_random_prediction_mnist(input_data)
                    elif thread_runner.runner_settings.dataset == "cifar10":
                        image, target_label = get_random_prediction_cifar_10(input_data)

                    p = t.predict(image)
                    pred = p['predicted'][0]
                    min_index, min_value = max(enumerate(pred), key=lambda x: x[1])
                    # print(pred)
                    if int(p['item']) == target_label:
                        print("Match", target_label)
                    else:
                        print("Prediction dont match:",  min_index, min_value, "label", target_label, p['item'])
                    sleep(5)

                if thread_runner.runner_settings.mode == "normal_runner":
                    if prediction_epoch == thread_runner.runner_settings.predict_epochs_swap:
                        if len(thread_runner.runner_settings.drifting_clients) == 0 or client_id in thread_runner.runner_settings.drifting_clients:
                            t.log("Start the DRIFT!")
                            start_drift = True
                            if drifter_function is None:
                                t.set_dataset(drifted_data, None)
                                current_dataset = drifted_data
                            else:
                                t.set_drifting(True)


                            prediction_epoch += 1
                            sleep(5)

                    if mode == "train":
                        t.log("Training round")
                        loss = t.train_round(False)
                        train_iterations_in_client += 1

                        if loss <= thread_runner.runner_settings.target_loss or (loss <= target_loss and train_iterations_in_client >= max_train_iterations):
                            t.log("We are trained! Going back to predicting...")
                            mode = "predict"

                    elif mode == "predict":
                        train_iterations_in_client = 0

                        if thread_runner.runner_settings.max_predict_epoch != 0 and prediction_epoch >= thread_runner.runner_settings.max_predict_epoch:
                            t.log("We are finished!")
                            break

                        image, target_label = get_random_prediction(current_dataset, thread_runner.runner_settings.dataset, drifter_function, prediction_epoch, start_drift)
                        p = t.predict(image, target_label, prediction_epoch)
                        t.log("Prediction ", p['item'], "should be ", target_label)
                        prediction_epoch += 1

                        # Absolutely how it should NOT be done, but fuck it we ballin
                        t.validation_data_loader.epoch = prediction_epoch
                        t.training_data_loader.epoch = prediction_epoch
                        # for _ in t.validation_data_loader:
                        #     pass
                        # for _ in t.training_data_loader:
                        #     pass
                        # sleep(0.1)
                        if thread_runner.runner_settings.check_testing_repeating > 0 and prediction_epoch % thread_runner.runner_settings.check_testing_repeating == 0:
                            loss, is_drifting = t.test_nn(prediction_epoch=prediction_epoch, type="sanity")
                            t.log("Sanity test of NN", loss, "with drifting", is_drifting)

                            if is_drifting:
                                print("Server says i am drifting! start drift mitigation...")
                                mode = "train"



                        if not thread_runner.runner_settings.disable_client_side_drift_detection:
                            if thread_runner.runner_settings.check_mode == "prediction":
                                list_of_last_results.insert(0, (1 if int(p['item']) == target_label else 0))

                                failures, all_tests = thread_runner.runner_settings.prediction_errors_count
                                list_of_last_results = list_of_last_results[:all_tests]

                                if len(list_of_last_results) == all_tests and all_tests - sum(list_of_last_results) >= failures:
                                    t.log("We have a failures! Double check with tests!")
                                    loss, is_drifting = t.test_nn(prediction_epoch=prediction_epoch)

                                    if loss > target_loss or is_drifting:
                                        t.log("Target loss not met ", loss, f" and server says {is_drifting}. Training!")
                                        mode = "train"
                                    else:
                                        t.log("is OK")

                                    list_of_last_results = []

                            if thread_runner.runner_settings.check_mode == "testing":
                                last_test_check += 1
                                if  last_test_check >= thread_runner.runner_settings.check_testing_repeating:
                                    t.log("Testing the predicted NN")
                                    last_test_check = 0
                                    loss, is_drifting = t.test_nn(prediction_epoch=prediction_epoch)
                                    if loss > target_loss or is_drifting:
                                        t.log("Target loss not met ", loss, f" or server says {is_drifting}. Training!")
                                        mode = "train"


            except KeyboardInterrupt:
                break

            except Exception as e:
                raise e
                pass

        t.log("Stopping client")
