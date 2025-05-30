import datetime
import json
import os
import threading
import shutil

import torch
from django.db import transaction
from django.db.models import StdDev, Avg, Min, Max, Sum
from rest_framework.decorators import api_view
from rest_framework.response import Response

from data_logger.models import DataTransferLog, TrainingLog, DriftingLogger, PredictionLog
from drift_detection.drift_detectors import SimpleAverageDriftDetection, check_average_drift_of_client
from neural_network.models import global_server_model

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('agg')


def report_usage(method, data_len, client_id, direction_to_server=True):
    DataTransferLog(source_method=method, data_transfer_len=data_len, target_source_client=client_id,
                    direction_to_server=direction_to_server).save()


def report_training(loss, client_id, epoch, server_epoch, training_time, last_communication_time,
                    last_whole_training_time, mode="training"):
    # print(training_time, last_communication_time, last_whole_training_time)
    TrainingLog(loss=loss, client_id=client_id, epoch=epoch, server_epoch=server_epoch, mode=mode,
                training_time=training_time, last_communication_time=last_communication_time,
                last_whole_training_time=last_whole_training_time).save()


lock = threading.Lock()

clients_amount = 0
current_client = 0

# Create your views here.
# request
sample_request_train = {
    "output": "list_tensor",
    "labels": "list_tensor",
    "local_epoch": "int",
    "client_id": "str"
}

drifting_suits = [
    SimpleAverageDriftDetection()
]



@api_view(["POST"])
def train(request):
    client_id = request.data["client_id"]
    last_communication_time = request.data['last_comm_time']
    last_whole_training_time = request.data['last_whole_training_time']

    local_epoch = request.data["local_epoch"]
    data_amount = sys.getsizeof(request.data)

    training_time = datetime.datetime.now()

    with transaction.atomic():

        with lock:
            return_data, loss = global_server_model.train_input(request.data['output'], request.data['labels'])
            time_length = (datetime.datetime.now() - training_time).total_seconds()

            data = {"gradients": return_data, "loss": loss}

            # if return_data.isnan().any() or return_data.isinf().any():
            #     print("THIS SHOULD NOT BE ERROR!")
            #     return Response({})
            report_training(loss, client_id, local_epoch, global_server_model.epoch, time_length, last_communication_time,
                            last_whole_training_time)

            report_usage("train", data_amount, client_id, direction_to_server=True)
            report_usage("train", sys.getsizeof(data), client_id, direction_to_server=False)

            global current_client

            current_client += 1
            if current_client >= clients_amount:
                current_client = 0

        return Response(data)


@api_view(["POST"])
def test(request):
    client_id = request.data["client_id"]
    local_epoch = request.data["local_epoch"]
    type_test = request.data['type']

    with transaction.atomic():

        report_usage(type_test, sys.getsizeof(request.data), client_id, direction_to_server=True)

        with lock:
            loss = global_server_model.test(request.data['output'], request.data['labels'])

        report_training(float(loss), client_id, local_epoch, global_server_model.epoch, 0, 0, 0, mode=type_test)

        if global_server_model.options['disable_server_side_drift_detection']:
            client_drifting = False
        else:
            print("Trying to drift detect", global_server_model.drift_detection_suite.drift_detection_run(), global_server_model.drift_detection_suite.drift_detection_class.get_regression())
            client_drifting = check_average_drift_of_client(client_id, global_server_model.options['server_zscore_deviation'], global_server_model.options['server_error_threshold'], global_server_model.options['server_filter_last_tests'])

        d = {"loss": loss, "client_drifting": client_drifting}

        report_usage(type_test, sys.getsizeof(d), client_id, direction_to_server=False)

    return Response(d)

@api_view(["POST"])
def mass_prediction_request(request):
    with transaction.atomic():
        client_id = request.data["client_id"]
        local_epoch = request.data["local_epoch"]
        input_data = request.data['input_data']
        with lock:
            items = []
            for input in input_data:
                predicted = global_server_model.predict(input)
                probabilities = torch.exp(predicted)
                probabilities[probabilities == float("Inf")] = 0
                items.append(torch.argmax(probabilities, dim=1).item())

    return Response({"data": items})

@api_view(["POST"])
def predict(request):
    with transaction.atomic():
        client_id = request.data["client_id"]
        local_epoch = request.data["local_epoch"]

        report_usage("predict", sys.getsizeof(request.data), client_id, direction_to_server=True)
        with lock:
            predicted = global_server_model.predict(request.data['output'])
            probabilities = torch.exp(predicted)
            probabilities[probabilities == float("Inf")] = 0

            data = {"predicted": probabilities, "item": torch.argmax(probabilities, dim=1).item()}

        # if "target_label" in request.data and request.data["target_label"] is not None:
        PredictionLog(client_id=client_id, client_epoch=request.data['local_epoch'], server_epoch=global_server_model.epoch, prediction_result=str(data['item']), expected_result=str(request.data['target_label'])).save()

        report_usage("predict", sys.getsizeof(data), client_id, direction_to_server=False)

    return Response(data)


@api_view(["POST"])
def report_client_nn_reset(request):
    client_id = request.data["client_id"]
    local_epochs = request.data["local_epochs"] if "local_epochs" in request.data else 0
    TrainingLog(mode="reset", client_id=client_id, epoch=local_epochs, server_epoch=global_server_model.epoch).save()


# @api_view(["POST"])
# def restart_runner(request):

#     return Response()


#
# - timeoverhead vs nr split layers and nr clients
# - communication overhead vs nr split layers and nr clients
# - also compare to centralized model the accuracy (see Figures 2, 3, and 4)

@api_view(["POST"])
def save_reports(request):

    details = request.data.get("details", {})

    if global_server_model.options['server_load_save_data'] and not global_server_model.options['load_only']:
        global_server_model.save(global_server_model.options['server_load_save_data'])

    qs = TrainingLog.objects.all().order_by('created_at')
    details["logs_timer"] = (qs.last().created_at - qs.first().created_at).total_seconds()
    details["server_model"] = str(global_server_model.model.model)
    details["server_optimiser"] = str(global_server_model.optimizer)
    details["server_loss_function"] = str(global_server_model.criterion)
    details["server_options"] = global_server_model.options

    results = TrainingLog.objects.aggregate(
        average_total_time=Avg("last_whole_training_time"),
        std_total_time=StdDev("last_whole_training_time"),
        average_comm_time=Avg("last_communication_time"),
        std_comm_time=StdDev("last_communication_time"),
        average_server_training_time=Avg("training_time"),
        std_server_training_time=StdDev("training_time"),
        minimal_loss=Min("loss"),
        avrg_loss=Avg("loss"),
        max_client_epoch=Max("epoch"),
        server_epoch=Max("server_epoch")
    )

    results_network = DataTransferLog.objects.aggregate(
        total_network_data = Sum("data_transfer_len"),
    )


    details['results'] = {**results, **results_network}


    folder = datetime.datetime.now().strftime("serverlogs/serverlogs-%Y%m%d%H%M%S")
    os.mkdir(folder)

    def j(file):
        return os.path.join(folder, file)

    db_file = "db.sqlite3"
    shutil.copy(os.path.abspath(db_file), j("db.sqlite3"))
    classes = [TrainingLog, DataTransferLog, PredictionLog, DriftingLogger]

    for cls in classes:
        df = pd.DataFrame(cls.objects.all().values())
        df.to_csv(j(cls.__name__ + ".csv"))


    with open(os.path.join(folder, "details.json"), "w") as f:
        json.dump(details, f, indent=4)

    #Save Model to file
    torch.save(global_server_model.model.state_dict(), j("server_model.pt"))

    # Save Training log to CSV
    df1 = pd.DataFrame(TrainingLog.objects.all().values())
    # df1.to_csv(j("training_log.csv"), index=False)

    df2 = pd.DataFrame(DataTransferLog.objects.all().values())
    # df2.to_csv(j("data_transfer_log.csv"), index=False)

    # df1.info()

    # Client Epoch to Server Epoch
    avg_epoch = df1[["epoch", "server_epoch"]].groupby(by="server_epoch").mean()

    # epoch loss with outlier detection
    epoch_los = df1[["loss", "server_epoch"]]
    q_hi = epoch_los['loss'].quantile(0.95)
    epoch_los = epoch_los[(epoch_los["loss"] < q_hi)]

    epoch_los.info()

    plt.plot(avg_epoch)
    plt.ylabel("Mean Client Epoch")
    plt.xlabel("Server Epoch")
    plt.savefig(j("avrg_epochs.png"))
    plt.close()

    plt.plot(epoch_los['server_epoch'], epoch_los['loss'])
    plt.ylabel("Loss")
    plt.xlabel("Server Epoch")
    plt.savefig(j("loss_epoch.png"))

    return Response({"server_data_folder": os.path.abspath(folder), "results": results})


@api_view(['POST'])
def prepare_running(request):
    # print("CO JEST??")
    global clients_amount, current_client
    options = request.data
    # print("test", options)
    global_server_model.options = request.data
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
    #     "client_learning_rate": 0.001,
#         "server_load_save_data": None,
    #     "client_load_directory": None,
    #     "reset_logs": True
    # }

    if options["reset_logs"]:
        TrainingLog.objects.all().delete()
        DataTransferLog.objects.all().delete()
        DriftingLogger.objects.all().delete()
        PredictionLog.objects.all().delete()

    if global_server_model.options['selected_model'] != global_server_model.model.model_number:
        global_server_model.reset_local_nn(options['selected_model'])

    clients_amount = options['clients']
    current_client = 0
    print(options)
    if options['server_load_save_data'] and not options['reset_nn'] and os.path.exists(os.path.join(options['server_load_save_data'], "server.pt")):
        print("Loading server file", os.path.join(options['server_load_save_data'], "server.pt"), "Created on", os.path.getctime(os.path.join(options['server_load_save_data'], "server.pt")))
        global_server_model.load(os.path.join(options['server_load_save_data'], "server.pt"))
        global_server_model.model.eval()
    else:
        print("Resetting server NN")
        global_server_model.reset_local_nn(options["selected_model"])


    global_server_model.reinit_optimiser(options["selected_model"], options.get("optimiser_overrides", {}).get("server_optimiser_parameters", None))

    return Response(options)


@api_view(['GET'])
def current_client(request):
    return Response({"current_client": current_client})
