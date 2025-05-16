import os
import random
import shutil
import signal
import threading
from dataclasses import dataclass, field
from datetime import datetime
from time import sleep
from typing import Literal, Tuple, Self

from django.utils.text import get_valid_filename

from client import run_client, ClientModel, TrainingSuite
from server_connection import ServerConnection
import json


@dataclass(frozen=True)
class RunnerArguments:

    #Run Description
    description: str = ""

    collected_folder_name: str = ""

    dataset: Literal["mnist", "cifar10"] = "mnist"

    #Number of Clients
    clients: int = 5
    #Run Mode
    mode: Literal["train", "predict_random", "normal_runner"] = "train"
    #selected model
    selected_model: int = 1


    #Optimiser learning rate for client
    client_learning_rate: float = 0.001
    #Arguments for server optimiser
    server_optimiser_options: dict = None

    #Synchronise clients (they go at the order)
    sync_mode: bool = False

    #Targets

    #Test run for max X seconds
    second_running: int = 0
    #Max Epoch per client
    client_epoch_limit: int = 0
    # Target Loss of any function
    target_loss: float = 0.1

    absolute_target: bool = True

    accuracy_target: float = 0.90
    accuracy_target_bool: bool = True

    # optimiser to use
    optimiser: Literal["sgd", "adamw"] = "sgd"

    #stop runner after X prediction epochs
    max_predict_epoch: int = 1000

    #should we wait for all clients to converge to target loss, or only one
    all_client_loss: bool = True


    #Data loading options
    #Target of shared server data
    server_load_save_data: str | None = None
    #Target of shared client data
    client_load_directory: str | None =  None
    #Only Load data, do not touch it back
    load_only: bool = False
    #Reset all logs before running NN
    reset_logs: bool =  True
    #Reset and randomise Neural Network before running NN
    reset_nn: bool = True


    #Drifting

    # Should we even run anything drifting related?
    start_drifting: bool = True
    #Disable server side drift detection
    disable_server_side_drift_detection: bool = False
    #disable client side drift detection
    disable_client_side_drift_detection: bool = False
    #after how many predict epochs should we start drifting?
    predict_epochs_swap:int =  30

    #What type of drifting to add?
    drift_type: Literal["dummy", "add_noise", "temporal_drift", "temporal_drift_client", "swap_domain", "half_clients_drift"] = "dummy"

    #What type of checking to execute?
    check_mode: Literal["prediction", "testing"] = "prediction"
    #For prediction mode, what should be the rate of errors (f.e. 2 errors in 10 prediction)
    prediction_errors_count: Tuple[int, int] = (2, 10)
    #For testing mode, after how many prediction epochs should we do a check?
    check_testing_repeating: int = 10

    #0 if make target_loss the basis for target
    #any number above - max percentage error above the baseline checked in a first iteration
    start_deviation_target: int = 0

    #Options for drifter function
    drifter_options: dict = None
    #What clients should drift? Type in numbers
    drifting_clients: list = None
    #You can define here, what clients should get what labels
    labels_filter: dict = None


    server_zscore_deviation: int = 5
    server_error_threshold: float = 0.2
    server_filter_last_tests: int = 50

    #Custom overrides for training data
    training_override: dict = None


    def construct_runner(self, *args: dict) -> Self:
        new_dict = {**self.__dict__}
        for d in args:
            new_dict = {**new_dict, **d}
        return RunnerArguments(**new_dict)


class SplitLearningRunner:

    def    __init__(self, runner_settings: RunnerArguments):
        self.runner_settings = runner_settings
        self.global_stop = False
        self.client_epochs_limit = runner_settings.client_epoch_limit
        self.seconds_running = runner_settings.second_running
        self.sync_mode = runner_settings.sync_mode
        self.folder = datetime.now().strftime("%Y%m%d%H%M%s")
        self.server_connection = ServerConnection("runner")
        self.target_loss = runner_settings.target_loss
        self.all_clients_loss = runner_settings.all_client_loss
        self.client_losses = {}
        self.client_learning_rate = runner_settings.client_learning_rate
        self.testing = True
        self.client_epochs = []

    def client_response(self, client_id, data):
        self.client_losses[client_id] = data['loss']
        if self.all_clients_loss:
            if all([x <= self.target_loss for x in self.client_losses.values()]):
                print("All loses are < los target")
                self.global_stop = True

        else:
            if self.target_loss > 0 and data['loss'] <= self.target_loss:
                print(f"We have a target loss for client! {client_id}")
                self.global_stop = True

    def add_epoch(self, epoch):
        self.client_epochs.append(epoch)

    def start_runner(self):
        print("Starting runner in 5 seconds")
        # sleep(5)
        self.server_connection.prepare_runner(self.runner_settings.__dict__)

        os.makedirs(self.folder, exist_ok=True)

        threads = []

        for i in range(self.runner_settings.clients):
            print("Starting Client", i)
            t = threading.Thread(target=run_client, args=[i, self])
            threads.append(t)
            t.start()
            sleep(0.5)

        def handler(signum, frame):
            print(signum)
            self.global_stop = True

        signal.signal(signal.SIGINT, handler)

        seconds_counter = 0
        while not self.global_stop or any([x.is_alive() for x in threads]):
            try:
                sleep(1)
                if all([not x.is_alive() for x in threads]):
                    print("All threads are done. Finishing the system")
                    self.global_stop = True

                if self.seconds_running > 0:
                    seconds_counter += 1
                    if seconds_counter % 15 == 0:
                        print("Passed", seconds_counter)
                    if seconds_counter >= self.seconds_running:
                        self.global_stop = True
                        print("STOP COUNTER REACHED TARGET TIME!!")

            except KeyboardInterrupt:
                print("STOP!")
                self.global_stop = True


        print("Stopping runners")

        sample_client = TrainingSuite(-1)

        if not (data_res := self.server_connection.save_report({
            "all_runner_options": self.runner_settings.__dict__,
            "client_model": str(sample_client.model.model),
            "client_optimizer": str(sample_client.optimizer),
            "clients": self.runner_settings.clients,
            "sync_mode": self.sync_mode,
            "seconds_running": self.seconds_running,
            "client_folder": self.folder,
            "target_loss": self.target_loss,
            "client_learning_rate": self.client_learning_rate,
        })):
            print("KURWA")


        self.collect_everything(data_res['server_data_folder'], self.runner_settings)
        data_res['results']["epochs_mean"] = sum(self.client_epochs) / len(self.client_epochs)

        return data_res['results']
        # while any(t.is_alive() for t in threads):
        #     sleep(1)

    def collect_everything(self, server, runner_settings):
        folder = datetime.now().strftime(f"{runner_settings.collected_folder_name}_%Y_%m_%d_%H_%M_%S")
        folder = "collected_runtime/"+get_valid_filename(folder)

        os.makedirs(folder, exist_ok=True)

        if server:
            shutil.copytree(server, os.path.join(folder, "server_data"))

        shutil.copytree(self.folder, os.path.join(folder, "client_data"))

        if self.runner_settings.server_load_save_data:
            j = os.path.join(r"/home/zgrate/mastersapp/SplitNN_Program", self.runner_settings.server_load_save_data)
            if os.path.exists(j):
                shutil.copytree(j, os.path.join(folder, "server_saved_model"))
            else:
                print("Probalby changed path of server! check")

        if self.runner_settings.client_load_directory:
            j = os.path.abspath(self.runner_settings.client_load_directory)
            print(j)
            if os.path.exists(j):
                shutil.copytree(j, os.path.join(folder, "client_saved_model"))
            else:
                print("Probalby changed path of client! check")

    def get_global_stop(self):
        return self.global_stop


if __name__ == "__main__":
    if True:
        default = RunnerArguments(server_optimiser_options={"lr": 0.001}, drifter_options={}, drifting_clients=[])

        # default = {
        #     "clients": 5,
        #     "sync_mode": False,
        #     "seconds_running": 0,
        #     "client_epochs_limit": 0,
        #     "target_loss": 1,
        #     "all_client_loss": True,
        #     "server_optimiser_options": {
        #         "lr": 0.001,
        #     },
        #     'drifter_options':{
        #     },
        #     "client_learning_rate": 0.001,
        #     "server_load_save_data": None,
        #     "client_load_directory": None,
        #     "load_only": False,
        #     "reset_logs": True,
        #     "reset_nn": True,
        #     "start_drifting": True,
        #     "mode": "train", #'train', 'predict_random',
        #     "check_mode": "prediction",
        #     "prediction_errors_count": (2 , 10),
        #     "predict_epochs_swap": 30,
        #     "check_testing_repeating": 10,
        #     "drifting_clients": [],
        #     "labels_filter": None,
        #     "drift_type": "add_noise",
        #     "disable_server_side_drift_detection": False,
        #     "disable_client_side_drift_detection": False
        # }
        clients = 5
        server_load_save_data = "server_test_2"
        client_load_directory = "client_test_2"

        def get_model_variant(model, clnts):
            return             (
                {
                'clients': clnts,
                'selected_model': model,
                'server_load_save_data': f"server_model_{clnts}_{model}",
                'client_load_directory': f"client_model_{clnts}_{model}",
            })

        model_variants = [
            # (0, 1),
            # (0, 2),
            (0, 4),
            #(0, 8),
            # (0, 16),
              # (0, 32),

            # (1, 1),
            # (1, 2),
            # (1, 4),
            # (1, 8),
            # (1, 16),
            # (1, 32),
            #
            # (2, 1),
            # (2, 2),
            # (2, 4)
            # (2, 4),
            # (2, 8),
            # (2, 16),
            # (2, 32),
            # # (2, 10),
            # (2, 12),
            #
            # (3, 4),
            # (3, 8),
            # (3, 16),
            # (3, 32)

            # (4, 4)
            # (6, 2)

        ]


        params = {
            "clients": clients,
            'server_load_save_data': server_load_save_data,
            'client_load_directory': client_load_directory,
        }

        disabled_drift_detection = {
            "disable_client_side_drift_detection": True,
            "disable_server_side_drift_detection": True,
        }

        enable_drift = {
            "predict_epochs_swap": 100
        }

        client_side_drift_detection = {
            "disable_client_side_drift_detection": False,
            "disable_server_side_drift_detection": True,
            "start_deviation_target": 0.2,
            'check_mode': "prediction",
            'prediction_errors_count': (2, 10),
            'check_testing_repeating': 20
        }

        server_side_drift_detection = {
            "disable_client_side_drift_detection": True,
            "disable_server_side_drift_detection": False,

            'server_zscore_deviation': 2,
            'server_error_threshold': 0.2,
            'server_filter_last_tests': 10,
            "second_running": 900
        }

        full_drift_detection = {
            **client_side_drift_detection,
            **server_side_drift_detection
        }

        drift_settings_add_noise = {
            **enable_drift,
            "drift_type": "add_noise",
            "drifter_options": {
                "noise_level": 0.1
            },
        }

        drift_settings_temporal_drift = {
            **enable_drift,
            "drift_type": "temporal_drift",
            "drifter_options": {
                "max_time_steps": 500,
                "start_epoch": 100,
                "max_time_epoch_drift": 700
            },

        }

        drift_settings_temporal_drift_client = {
            **enable_drift,
            "drift_type": "temporal_drift_client",
            "drifter_options": {
                "max_time_steps": 500,
                "start_epoch": 100,
                "max_time_epoch_drift": 700
            },

        }

        fresh_run = {
            "reset_logs": True,
            "reset_nn": True,
        }

        load_run = {
            "reset_logs": True,
            "reset_nn": False,
            "load_only": True
        }

        zero_training = {
            "optimiser": "sgd",
            "server_optimiser_options": {"lr": 0.0001},
            # "load_only": True,
        }

        training_settings = {
            "reset_nn": False,
            "reset_logs": True,
            "load_only": False,
            'target_loss': 0.4,
            "mode": "train",
            # 'second_running': 1800
        }

        test_settings = {
            **load_run,
            'target_loss': 0.4,
            "mode": "normal_runner",
            "predict_epochs_swap": 100
        }

        debug_test_settings = {
            **load_run,
            "mode": "test"
        }

        clients_base = {"clients": 2}


        settings = [
            #Main Test

            # default.construct_runner(params, disabled_drift_detection, fresh_run, drift_settings_add_noise, shared_settings, training_settings, clients_base),
            # default.construct_runner(params, disabled_drift_detection, load_run, drift_settings_add_noise,  shared_settings, clients_base),
            # default.construct_runner(params, disabled_drift_detection, load_run, drift_settings_temporal_drift, shared_settings, clients_base)

            # {"clients": 2, **default},

            #Default 5 client run with model 1 and no drift detection
            # default.construct_runner(params, disabled_drift_detection, fresh_run, drift_settings_add_noise, shared_settings, {'target_loss': 0.2, "mode": "train"}),


            #Add noise
            # default.construct_runner(params, disabled_drift_detection, load_run, drift_settings_add_noise,  shared_settings),

            #Temporal
            # default.construct_runner(params, disabled_drift_detection, load_run, drift_settings_temporal_drift, shared_settings)

            # {**default, **params, **disabled_drift_detection, 'target_loss': 0.1, 'reset_logs': True, 'reset_nn': True, "mode": "normal_runner"},
            # {**default, **params, 'target_loss': 0.1, 'reset_logs': False, "load_only": False, 'reset_nn': False, "mode": "train"},
            # {**default, **params, 'reset_logs': False, 'reset_nn': False, "load_only": True, "mode": "predict_random"},

            # Add Noise Drift and only 2 clients drift, with label separation
            # {
            #     **default,
            #     **params,
            #     "drift_type": "add_noise",
            #     "target_loss": 0.2,
            #     "reset_logs": False,
            #     "reset_nn": False,
            #     "load_only": True,
            #     "mode": "normal_runner",
            #     'server_load_save_data': 'server_test_1',
            #     'client_load_directory': 'client_test_1',
            #       'drifter_options':
            #         {
            #            'noise_level': 0.2
            #         },
            #     "drifting_clients": [0, 1],
            #     "labels_filter": {
            #         0: [0, 1, 2],
            #         1: [3, 4],
            #         2: [5, 6],
            #         3: [7, 8],
            #         4: [9, 0],
            #     }
            # }

            # Add Noise Drift
            # default.construct_runner(params, {
            #     "drift_type": "add_noise",
            #     "target_loss": 0.2,
            #     "reset_logs": False,
            #     "reset_nn": False,
            #     "load_only": True,
            #     "mode": "normal_runner",
            #     'server_load_save_data': 'server_test_1',
            #     'client_load_directory': 'client_test_1',
            #     'drifter_options':
            #         {
            #             'noise_level': 0.2
            #         }
            # })

            #Temporal Drift
            # {
            #     **default,
            #     **params,
            #     'desc': "Normal Running with temporal drift from 20th iteration",
            #     "drift_type": "temporal_drift",
            #     "target_loss": 0.2,
            #     "reset_logs": False,
            #     "reset_nn": False,
            #     "load_only": True,
            #     "mode": "normal_runner",
            #     'server_load_save_data': 'server_test_1',
            #     'client_load_directory': 'client_test_1',
            #     'drifter_options': {
            #         'max_time_steps': 200,
            #         'start_epoch': 20
            #     },
            # }


            # Domain Shift
            # {**default, **params, 'target_loss': 0.2, 'reset_logs': True, 'reset_nn': True, "load_only": False, "mode": "train"},
            # {**default, **params, 'target_loss': 0.2, 'reset_logs': True, 'reset_nn': True, "load_only": False, "mode": "normal_runner"},
            # {
            #     **default,
            #     **params,
            #     'desc': "Normal Running with domain Shift",
            #     'target_loss': 0.2,
            #     'reset_logs': False,
            #     'reset_nn': False,
            #     "load_only": True,
            #     "mode": "normal_runner",
            #     'drift_type': "swap_domain",
            #     'server_load_save_data': 'server_test_2',
            #     'client_load_directory': 'client_test_2'
            # },

            # {"clients": 4, **default},
            # {"clients": 5, **default},
            # {"clients": 6, **default},
            # {"clients": 7, **default},
            # {"clients": 8, **default},
            # {"clients": 9, **default},
            # {"clients": 10, **default},
        ]
        # settings = [
        #     default.construct_runner({"description": str((2, 2))}, params, disabled_drift_detection, drift_settings_add_noise,
        #                              training_settings, get_model_variant(2,2))
        # ]
        #

        #All Training
        # settings = [default.construct_runner({"description": str(x), "dataset": "cifar10"}, training_settings, zero_training, get_model_variant(*x)) for x in model_variants]
        settings = [default.construct_runner({"description": str(x), "dataset": "mnist"}, training_settings, zero_training, get_model_variant(*x)) for x in model_variants]


        #settings = settings + [default.construct_runner({"description": str(x)}, debug_test_settings, get_model_variant(*x)) for x in model_variants]
        # print(len(settings), settings)
        #Add noise moment
        # settings = [
        #     default.construct_runner({"collected_folder_name": f"Noisy Model {x[0]} Clients {x[1]}", "description": "Noisy" + str(x)}, disabled_drift_detection, drift_settings_add_noise, test_settings, get_model_variant(*x)) for x in model_variants
        # ]

        # settings = [
        #     default.construct_runner({"collected_folder_name": f"Temporal Model {x[0]} Clients {x[1]}", "description": "Temporal" + str(x)}, disabled_drift_detection, drift_settings_temporal_drift, test_settings, get_model_variant(*x)) for x in model_variants
        # ]

        x = (1, 6)
        # settings = [
        #     default.construct_runner({"collected_folder_name": f"Noise Add Model {x[0]} Clients {x[1]}", "description": "Noise Add" + str(x)}, client_side_drift_detection, drift_settings_add_noise, test_settings, get_model_variant(*x))
        # ]

        # settings = [
        #     default.construct_runner({"collected_folder_name": f"temportal Add Model {x[0]} Clients {x[1]}", "description": "Noise Add" + str(x)}, client_side_drift_detection, drift_settings_temporal_drift, test_settings, get_model_variant(*x))
        # ]

        # settings = [
        #     default.construct_runner({"collected_folder_name": f"server side temportal Add Model {x[0]} Clients {x[1]}", "description": "Noise Add" + str(x)}, server_side_drift_detection, drift_settings_temporal_drift, test_settings, get_model_variant(*x))
        # ]

        # settings = [
        #     default.construct_runner({"collected_folder_name": f"server side temportal Add Model {x[0]} Clients {x[1]}", "description": "Noise Add" + str(x)}, server_side_drift_detection, drift_settings_temporal_drift_client, test_settings, get_model_variant(*x))
        # ]
        if False:
            while True:
                queue_file = "queue.json"
                errors_file = "errors.json"
                if not os.path.exists(queue_file):
                    shutil.copy("output_zero.json", queue_file)

                with open(queue_file) as f:
                    queue = json.load(f)

                settings = [RunnerArguments(**x) for x in queue]

                print(settings)
                for i in range(len(settings)):
                    setting = settings.pop(0)
                    if setting.clients == 2:
                        continue
                    print("Running", setting, setting.description)
                    try:
                        SplitLearningRunner(setting).start_runner()
                    except Exception as e:
                        print("Error running file! Reporting errorr")
                        with open("errors_file", "a") as f:
                            f.write(str(e) + "\n")
                            json.dump(setting.__dict__, f)
                    print("Runner Finished! You can stop here or wait for next run")
                    print(f"left {len(settings)}")
                    with open("queue.json", "w") as f:
                        json.dump([x.__dict__ for x in settings], f)
                    print("Saved queue!")

                    sleep(10)
                print("Finished the run! repeating...")
                os.remove("queue.json")
                # exit()
                # shutil.copy("output_target.json", queue_file)
                sleep(60)

        for i in range(0, 1):
            for setting in settings:
                print("Running", setting, setting.description)
                SplitLearningRunner(setting).start_runner()
                print("Runner Finished! You can stop here or wait for next run")
                sleep(10)
        exit(0)

    setting = {
        "clients": 5,
        "sync_mode": False,
        "seconds_running": 0,
        "client_epochs_limit": 0,
        "target_loss": 0.5,
        "all_client_loss": True,
        "client_learning_rate": 0.001,
        "server_learning_rate": 0.001,
    }
    print("Running runner with settings")
    print(setting)
    SplitLearningRunner(setting).start_runner()
