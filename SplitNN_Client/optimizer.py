import datetime
import os.path

import optuna
import logging
from typing import Dict, Union, List
import random

from SplitNN_Client.runner import RunnerArguments, SplitLearningRunner
from training_generator import disabled_drift_detection, drift_settings_temporal_drift, test_settings, \
    drift_settings_add_noise, drift_settings_temporal_drift_client

# Configuration parameters and their default values
DEFAULT_PARAMS = {
    "check_testing_repeating": 10,
    "prediction_errors_count": [2, 20],
    "start_deviation_target": 0.2,
    "server_zscore_deviation": 1.55,
    "server_error_threshold": 0.2,
    "server_filter_last_tests": 20
}

model_variants = [
    # (1, 1),
    (1, 2),
    (1, 4),
    (1, 8),
    (1, 16),
    (1, 32),

    # (2, 1),
    (2, 2),
    (2, 4),
    (2, 8),
    (2, 16),
    (2, 32),
]

def get_model_variant(model, clnts):
    return             (
        {
        'clients': clnts,
        'selected_model': model,
        'server_load_save_data': os.path.join("server_models", f"server_model_{clnts}_{model}"),
        'client_load_directory': os.path.join("client_models", f"client_model_{clnts}_{model}"),
    })

default = RunnerArguments(server_optimiser_options={"lr": 0.001}, drifter_options={}, drifting_clients=[])

def generate_no_drift(params: Dict = DEFAULT_PARAMS):
    no_drift = []
    # target runner
    for x in model_variants:
        model, client = x
        no_drift.append(default.construct_runner(
            {"collected_folder_name": f"No Drift Detection  {x[0]} Clients {x[1]} temporal",
             "description": "No Drift" + str(x)}, disabled_drift_detection, drift_settings_temporal_drift,
            test_settings, get_model_variant(model, client), params))
        no_drift.append(default.construct_runner(
            {"collected_folder_name": f"No Drift Detection   {x[0]} Clients {x[1]} noise",
             "description": "No Drift" + str(x)}, disabled_drift_detection, drift_settings_add_noise, test_settings,
            get_model_variant(model, client), params))
        no_drift.append(default.construct_runner(
            {"collected_folder_name": f"No Drift Detection   {x[0]} Clients {x[1]} temportal client",
             "description": "No Drift" + str(x)}, disabled_drift_detection, drift_settings_temporal_drift_client,
            test_settings, get_model_variant(model, client), params))

    return no_drift

def generate_drifts(params: Dict = DEFAULT_PARAMS):

    target_runner = []



    for x in model_variants:
        model, client = x
        target_runner.append(default.construct_runner(
            {"collected_folder_name": f"Drift detect drift  {x[0]} Clients {x[1]} temporal",
             "description": "Drift detect" + str(x)}, drift_detection_hiperparameters_model_settings,
            drift_settings_temporal_drift, test_settings, get_model_variant(model, client), params))
        target_runner.append(default.construct_runner(
            {"collected_folder_name": f"Drift detect drift  {x[0]} Clients {x[1]} noise",
             "description": "Drift detect" + str(x)}, drift_detection_hiperparameters_model_settings,
            drift_settings_add_noise, test_settings, get_model_variant(model, client), params))
        target_runner.append(default.construct_runner(
            {"collected_folder_name": f"Drift detect drift  {x[0]} Clients {x[1]} temportal client",
             "description": "Drift detect" + str(x)}, drift_detection_hiperparameters_model_settings,
            drift_settings_temporal_drift_client, test_settings, get_model_variant(model, client), params))

    return target_runner

one_models = [
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 8),
    (1, 16),
    (1, 32),
]

training_settings = {
    "reset_nn": True,
    "reset_logs": True,
    # "load_only": True,
    'target_loss': 0.4,
    "mode": "train",
    'second_running': 1800
}

zero_training = {
    "optimiser": "sgd",
    # "server_optimiser_options": {"lr": 0.0001},
    # "load_only": True,
}

def training_objective(trial: optuna.Trial) -> float:

    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        "momentum": trial.suggest_float("momentum", 0.0, 1.0),
        "batch_size": trial.suggest_int("batch_size", 32, 128, step=32),
    }

    data = {
        "training_override": {
            "optimiser_parameters":{
                "lr": params["lr"],
                "momentum": params["momentum"]
            },
            "server_optimiser_parameters": {
                "lr": params["lr"],
                "momentum": params["momentum"]
            },
            "batch_size": params["batch_size"]
        }
    }

    res = []
    settings = [default.construct_runner({"description": str(x), "dataset": "mnist"}, training_settings, zero_training,
                              get_model_variant(*x), data) for x in one_models]

    time_calc = []
    for setting in settings:
        startTime = datetime.datetime.now()
        results = SplitLearningRunner(setting).start_runner()
        res.append(results['epochs_mean'])
        print(results['epochs_mean'])
        endTime = datetime.datetime.now()
        time_calc.append((endTime - startTime).total_seconds())
        # exit(0)



    return sum(time_calc) / len(time_calc)





def evaluate_training_params(params: Dict) -> float:
    """Evaluate parameters by constructing runner arguments."""
    a = []

    for x in random.sample(generate_drifts(params), 7):
        startTime = datetime.datetime.now()
        results = SplitLearningRunner(x).start_runner()
        endTime = datetime.datetime.now()
        a.append((endTime - startTime).total_seconds())

    return sum(a) / len(a)

def objective(trial: optuna.Trial) -> float:
    # Define parameter ranges for optimization
    params = {
        "check_testing_repeating": trial.suggest_int("check_testing_repeating", 5, 20),
        "prediction_errors_count": (trial.suggest_int("prediction_errors_count_min", 1, 10), trial.suggest_int("prediction_errors_count_max", 11, 30)),
        "start_deviation_target": trial.suggest_float("start_deviation_target", 0.1, 0.5),
        "server_zscore_deviation": trial.suggest_float("server_zscore_deviation", 1.0, 2.0),
        "server_error_threshold": trial.suggest_float("server_error_threshold", 0.1, 0.5),
        "server_filter_last_tests": trial.suggest_int("server_filter_last_tests", 10, 50)
    }

    # TODO: Add your evaluation metric here
    score = evaluate_parameters(params)

    return score


def evaluate_parameters(params: Dict) -> float:
    """Evaluate parameters by constructing runner arguments."""
    a = []

    for x in random.sample(generate_drifts(params), 7):
        startTime = datetime.datetime.now()
        results = SplitLearningRunner(x).start_runner()
        endTime = datetime.datetime.now()
        a.append((endTime - startTime).total_seconds())

    return sum(a) / len(a)

def optimize_training_params(n_trials: int = 100) -> Dict:
    study = optuna.create_study(storage="sqlite:///db_optuna.sqlite3",direction="minimize")
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler())

    study.optimize(training_objective, n_trials=n_trials)

    print(study.best_params)

    return study.best_params


def optimize_parameters(n_trials: int = 100) -> Dict:
    study = optuna.create_study(storage="sqlite:///db_optuna.sqlite3",direction="minimize")

    # Configure logging
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler())

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Get best parameters
    best_params = study.best_params
    best_params["prediction_errors_count"] = [
        best_params.pop("prediction_errors_count_min"),
        best_params.pop("prediction_errors_count_max")
    ]

    return best_params


if __name__ == "__main__":
    optimized_params = optimize_training_params()
    print("Optimized parameters:", optimized_params)
