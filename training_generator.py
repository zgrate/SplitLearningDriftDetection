import pandas

import os
import random
import shutil
import signal
import threading
from dataclasses import dataclass, field
from datetime import datetime
from time import sleep
from typing import Literal, Tuple, Self


@dataclass(frozen=True)
class RunnerArguments:

    #Run Description
    description: str = ""

    collected_folder_name: str = ""

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
    second_running: int = 900
    #Max Epoch per client
    client_epoch_limit: int = 0
    # Target Loss of any function
    target_loss: float = 0.1

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
    drift_type: Literal["dummy", "add_noise", "temporal_drift", "temporal_drift_client", "swap_domain"] = "dummy"

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


    server_zscore_deviation: int = 2
    server_error_threshold: float = 0.2
    server_filter_last_tests: int = 50

    def construct_runner(self, *args: dict) -> Self:
        new_dict = {**self.__dict__}
        for d in args:
            new_dict = {**new_dict, **d}
        return RunnerArguments(**new_dict)



model_variants = [

    # (0, 1),
    # (0, 2),
    # (0, 4),
    # (0, 8),
    # (0, 16),
    # (0, 32)

   # (1, 2),
    (1, 4),
    (1, 8),
    # (1, 12),
    (1, 16),
    (1, 32),

    #(2, 2),
    (2, 4),
    (2, 8),
    (2, 16),
    (2, 32),

    # (2, 12),

  #  (3, 2),
    # (3, 4),
    # (3, 8),
    # (3, 16),
    # (3, 32),
    # (3, 12),


]

#"selected_model,clients,disable_client_side_drift_detection,disable_server_side_drift_detection,drift_type,check_mode,check_testing_repeating,prediction_errors_count,start_deviation_target"

columns = ['selected_model', 'clients', 'disable_client_side_drift_detection', 'disable_server_side_drift_detection', 'drift_type', 'check_mode', 'check_testing_repeating', 'prediction_errors_count', 'start_deviation_target',   'server_zscore_deviation', 'server_error_threshold', 'server_filter_last_tests']

default = RunnerArguments(server_optimiser_options={"lr": 0.001}, drifter_options={}, drifting_clients=[])
def get_model_variant(model, clnts):
    return             (
        {
        'clients': clnts,
        'selected_model': model,
        'server_load_save_data': f"server_model_{clnts}_{model}",
        'client_load_directory': f"client_model_{clnts}_{model}",
    })




disabled_drift_detection = {
    "disable_client_side_drift_detection": True,
    "disable_server_side_drift_detection": True,
}

enable_drift = {
    "predict_epochs_swap": 100
}

client_side_drift_detection_prediction = {
    "disable_client_side_drift_detection": False,
    "disable_server_side_drift_detection": True,
    "start_deviation_target": 0.2,
    'check_mode': "prediction",
    'prediction_errors_count': (2, 10),
    'check_testing_repeating': 20
}


client_side_drift_detection_testing = {
    "disable_client_side_drift_detection": False,
    "disable_server_side_drift_detection": True,
    'check_mode': "testing",
    'check_testing_repeating': 20,
    "start_deviation_target": 0.2,
}

server_side_drift_detection = {
    "disable_client_side_drift_detection": True,
    "disable_server_side_drift_detection": False,

    'server_zscore_deviation': 5,
    'server_error_threshold': 0.2,
    'server_filter_last_tests': 10,
    "second_running": 1800
}

# full_drift_detection = {
#     **client_side_drift_detection,
#     **server_side_drift_detection
# }

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

drift_settings_half_clients = {
    **enable_drift,
    "drift_type": "half_clients_drift",
    "drifter_options": {
        "noise_level": 0.15
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

training_settings = {
    **fresh_run,
    'target_loss': 0.4,
    "mode": "train",
}

test_settings = {
    **load_run,
    'target_loss': 0.4,
    "mode": "normal_runner",
    "predict_epochs_swap": 100
}


drift_detection_hiperparameters_model_settings = {
    "disable_client_side_drift_detection": False,
    "disable_server_side_drift_detection": False,

    'check_mode': "prediction",
    "start_deviation_target": 0.2,
    'prediction_errors_count': (2, 20),
    'check_testing_repeating': 10,

    'server_zscore_deviation': 1.55,
    'server_error_threshold': 0.2,
    'server_filter_last_tests': 20,

    "second_running": 1800

}

no_drift_only_test = {
    "disable_client_side_drift_detection": True,
    "disable_server_side_drift_detection": True,
    "drift_type": "no_drift",

}

no_drift = []

#target runner
for x in model_variants:
    model, client = x
    no_drift.append(default.construct_runner({"collected_folder_name": f"No Drift Detection  {x[0]} Clients {x[1]} temporal", "description": "No Drift" + str(x)}, disabled_drift_detection, drift_settings_temporal_drift, test_settings, get_model_variant(model, client)))
    no_drift.append(default.construct_runner({"collected_folder_name": f"No Drift Detection   {x[0]} Clients {x[1]} noise", "description": "No Drift" + str(x)}, disabled_drift_detection, drift_settings_add_noise,test_settings, get_model_variant(model, client)))
    no_drift.append(default.construct_runner({"collected_folder_name": f"No Drift Detection   {x[0]} Clients {x[1]} temportal client", "description": "No Drift" + str(x)}, disabled_drift_detection, drift_settings_temporal_drift_client,test_settings, get_model_variant(model, client)))

print(len(no_drift))

#default runner with no drift detection
target_runner = []
for x in model_variants:
    model, client = x
    target_runner.append(default.construct_runner({"collected_folder_name": f"Drift detect drift  {x[0]} Clients {x[1]} temporal", "description": "Drift detect" + str(x)}, drift_detection_hiperparameters_model_settings, drift_settings_temporal_drift, test_settings, get_model_variant(model, client)))
    target_runner.append(default.construct_runner({"collected_folder_name": f"Drift detect drift  {x[0]} Clients {x[1]} noise", "description": "Drift detect" + str(x)}, drift_detection_hiperparameters_model_settings, drift_settings_add_noise,test_settings, get_model_variant(model, client)))
    target_runner.append(default.construct_runner({"collected_folder_name": f"Drift detect drift  {x[0]} Clients {x[1]} temportal client", "description": "Drift detect" + str(x)}, drift_detection_hiperparameters_model_settings, drift_settings_temporal_drift_client,test_settings, get_model_variant(model, client)))

#half drift
half_drift = []
for x in model_variants:
    model, client = x
    half_drift.append(default.construct_runner({"collected_folder_name": f"Half Drift detect drift  {x[0]} Clients {x[1]}", "description": "Half Drift detect" + str(x)}, drift_detection_hiperparameters_model_settings, drift_settings_half_clients, test_settings, get_model_variant(model, client)))
    half_drift.append(default.construct_runner({"collected_folder_name": f"Half No Drift Detection   {x[0]} Clients {x[1]} ", "description": "Half No Drift" + str(x)}, disabled_drift_detection, drift_settings_half_clients, test_settings, get_model_variant(model, client)))

#no drifting reference
no_drifting_ref = []
for x in model_variants:
    model, client = x
    no_drifting_ref.append(default.construct_runner({"collected_folder_name": f"No drifting reference  {x[0]} Clients {x[1]} temporal", "description": "No drifting reference" + str(x)}, no_drift_only_test, test_settings, get_model_variant(model, client)))

#zero model
zero_model = []
for x in model_variants:
    model, client = x
    zero_model.append(default.construct_runner({"collected_folder_name": f"zero detect drift  {x[0]} Clients {x[1]} temporal", "description": "Drift detect" + str(x)}, drift_detection_hiperparameters_model_settings, drift_settings_temporal_drift, test_settings, get_model_variant(model, client)))
    zero_model.append(default.construct_runner({"collected_folder_name": f"zero detect drift  {x[0]} Clients {x[1]} noise", "description": "Drift detect" + str(x)}, drift_detection_hiperparameters_model_settings, drift_settings_add_noise,test_settings, get_model_variant(model, client)))
    zero_model.append(default.construct_runner({"collected_folder_name": f"zero detect drift  {x[0]} Clients {x[1]} temportal client", "description": "Drift detect" + str(x)}, drift_detection_hiperparameters_model_settings, drift_settings_temporal_drift_client,test_settings, get_model_variant(model, client)))

    zero_model.append(default.construct_runner({"collected_folder_name": f"zero No Drift Detection  {x[0]} Clients {x[1]} temporal", "description": "No Drift" + str(x)}, disabled_drift_detection, drift_settings_temporal_drift, test_settings, get_model_variant(model, client)))
    zero_model.append(default.construct_runner({"collected_folder_name": f" zeroNo Drift Detection   {x[0]} Clients {x[1]} noise", "description": "No Drift" + str(x)}, disabled_drift_detection, drift_settings_add_noise,test_settings, get_model_variant(model, client)))
    zero_model.append(default.construct_runner({"collected_folder_name": f" zeroNo Drift Detection   {x[0]} Clients {x[1]} temportal client", "description": "No Drift" + str(x)}, disabled_drift_detection, drift_settings_temporal_drift_client,test_settings, get_model_variant(model, client)))

    zero_model.append(default.construct_runner({"collected_folder_name": f"zero No drifting reference  {x[0]} Clients {x[1]} temporal", "description": "No drifting reference" + str(x)}, no_drift_only_test, test_settings, get_model_variant(model, client)))





all_together = [x.__dict__ for x in half_drift]

print(len(all_together))

import json
with open("output_half_drift.json", mode="w") as output:
    json.dump(all_together, output)
    

#client drift on testing mode
# client_drift_detection_testing_list  = []

# # start_deviation_target = [0.2, 0.1, 0.4]
# start_deviation_target = [0.2]
# check_testing_repeating = [10]
# # check_testing_repeating = [10, 20, 50]




# for x in model_variants:
#     model, client = x

#     for dev_target in start_deviation_target:
#         for test_target in check_testing_repeating:
#             client_drift_detection_testing_list.append(default.construct_runner({"collected_folder_name": f"Client {x[0]} Clients {x[1]}", "description": "No Drift" + str(x)}, client_side_drift_detection_testing, drift_settings_add_noise, test_settings, get_model_variant(model, client), {"start_deviation_target": dev_target, "check_testing_repeating": test_target}))
#             client_drift_detection_testing_list.append(default.construct_runner({"collected_folder_name": f"Client {x[0]} Clients {x[1]}", "description": "No Drift" + str(x)}, client_side_drift_detection_testing, drift_settings_temporal_drift, test_settings, get_model_variant(model, client), {"start_deviation_target": dev_target, "check_testing_repeating": test_target}))
#             client_drift_detection_testing_list.append(default.construct_runner({"collected_folder_name": f"Client {x[0]} Clients {x[1]}", "description": "No Drift" + str(x)}, client_side_drift_detection_testing, drift_settings_temporal_drift_client, test_settings, get_model_variant(model, client), {"start_deviation_target": dev_target, "check_testing_repeating": test_target}))


# #client drift on prediction mode
# client_drift_detection_prediction_list  = []

# start_deviation_target = [0.2, 0.1, 0.4]
# check_testing_repeating = [10, 20, 50]


# prediction_errors_count = [(2,20)]

# for x in model_variants:
#     model, client = x

#     for dev_targe78t in start_deviation_target:
#         for test_target in check_testing_repeating:
#             for pred_count in prediction_errors_count:
#                 client_drift_detection_prediction_list.append(default.construct_runner({"collected_folder_name": f"Client 2 {x[0]} Clients {x[1]}", "description": "No Drift" + str(x)}, client_side_drift_detection_prediction, drift_settings_add_noise, test_settings, get_model_variant(model, client), {"start_deviation_target": dev_target, "check_testing_repeating": test_target, 'prediction_errors_count': pred_count}))
#                 client_drift_detection_prediction_list.append(default.construct_runner({"collected_folder_name": f"Client 2 {x[0]} Clients {x[1]}", "description": "No Drift" + str(x)}, client_side_drift_detection_prediction, drift_settings_temporal_drift, test_settings, get_model_variant(model, client), {"start_deviation_target": dev_target, "check_testing_repeating": test_target, 'prediction_errors_count': pred_count}))
#                 client_drift_detection_prediction_list.append(default.construct_runner({"collected_folder_name": f"Client 2 {x[0]} Clients {x[1]}", "description": "No Drift" + str(x)}, client_side_drift_detection_prediction, drift_settings_temporal_drift_client, test_settings, get_model_variant(model, client), {"start_deviation_target": dev_target, "check_testing_repeating": test_target, 'prediction_errors_count': pred_count}))

# prediction_serveR_model = []

# server_zscore_deviation = [1.5]
# server_error_threshold = [0.2]
# server_filter_last_tests = [20]

# for x in model_variants:
#     model, client = x
#     for dev_zscore_dev in server_zscore_deviation:
#         for dev_server_error in server_error_threshold:
#             for dev_filter_test in server_filter_last_tests:
#                 prediction_serveR_model.append(default.construct_runner({"collected_folder_name": f"server dev Client 2 {x[0]} Clients {x[1]}", "description": "No Drift" + str(x)}, server_side_drift_detection, drift_settings_temporal_drift_client, test_settings, get_model_variant(model, client), {"server_zscore_deviation": dev_zscore_dev, "server_error_threshold": dev_server_error, 'server_filter_last_tests': dev_filter_test}))


# print(len(prediction_serveR_model))

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



