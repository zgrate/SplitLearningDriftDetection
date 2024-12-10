import datetime
import json
import os
import threading

import torch
from django.db.models import StdDev, Avg, Min, Max, Sum
from rest_framework.decorators import api_view
from rest_framework.response import Response

from data_logger.models import DataTransferLog, TrainingLog
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


@api_view(["POST"])
def train(request):
    client_id = request.data["client_id"]
    last_communication_time = request.data['last_comm_time']
    last_whole_training_time = request.data['last_whole_training_time']

    local_epoch = request.data["local_epoch"]
    data_amount = sys.getsizeof(request.data)

    training_time = datetime.datetime.now()

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

    report_usage("test", request.data, client_id, direction_to_server=True)

    with lock:
        loss = global_server_model.test(request.data['output'], request.data['labels'])

    report_usage("test", loss, client_id, direction_to_server=False)

    return Response({"loss": loss})


@api_view(["POST"])
def predict(request):
    client_id = request.data["client_id"]

    report_usage("predict", request.data, client_id, direction_to_server=True)
    with lock:
        data = {"predicted": global_server_model.predict(request.data['output'])}

    report_usage("predict", data, client_id, direction_to_server=False)

    return Response(data)


@api_view(["POST"])
def report_client_nn_reset(request):
    client_id = request.data["client_id"]
    local_epochs = request.data["local_epochs"] if "local_epochs" in request.data else 0
    TrainingLog(mode="reset", client_id=client_id, epoch=local_epochs, server_epoch=global_server_model.epoch).save()


@api_view(["POST"])
def restart_runner(request):

    TrainingLog.objects.all().delete()
    DataTransferLog.objects.all().delete()
    global_server_model.reset_local_nn()
    return Response()


#
# - timeoverhead vs nr split layers and nr clients
# - communication overhead vs nr split layers and nr clients
# - also compare to centralized model the accuracy (see Figures 2, 3, and 4)

@api_view(["POST"])
def save_reports(request):

    details = request.data.get("details", {})
    print(details)

    qs = TrainingLog.objects.all().order_by('created_at')
    details["logs_timer"] = (qs.last().created_at - qs.first().created_at).total_seconds()
    details["server_model"] = str(global_server_model.model.model)
    details["server_optimiser"] = str(global_server_model.optimizer)
    details["server_loss_function"] = str(global_server_model.criterion)

    results = TrainingLog.objects.aggregate(
        average_total_time=Avg("last_whole_training_time"),
        std_total_time=StdDev("last_whole_training_time"),
        average_comm_time=Avg("last_communication_time"),
        std_comm_time=StdDev("last_communication_time"),
        average_server_training_time=Avg("training_time"),
        std_server_training_time=StdDev("training_time"),
        minimal_loss=Min("loss"),
        max_client_epoch=Max("epoch"),
        server_epoch=Max("server_epoch")
    )

    results_network = DataTransferLog.objects.aggregate(
        total_network_data = Sum("data_transfer_len"),
    )


    nn_models.ClientModel()


    details['results'] = {**results, **results_network}


    folder = datetime.datetime.now().strftime("serverlogs/serverlogs-%Y%m%d%H%M%S")
    os.mkdir(folder)

    def j(file):
        return os.path.join(folder, file)

    with open(os.path.join(folder, "details.json"), "w") as f:
        json.dump(details, f, indent=4)

    #Save Model to file
    torch.save(global_server_model.model.state_dict(), j("server_model.pt"))

    # Save Training log to CSV
    df1 = pd.DataFrame(TrainingLog.objects.all().values())
    df1.to_csv(j("training_log.csv"), index=False)

    df2 = pd.DataFrame(DataTransferLog.objects.all().values())
    df2.to_csv(j("data_transfer_log.csv"), index=False)

    df1.info()

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

    return Response({})


@api_view(['POST'])
def prepare_running(request):
    global clients_amount, current_client

    clients_amount = request.data['clients_amount']
    current_client = 0
    global_server_model.reinit_optimiser(**request.data['server_optimiser_options'])
    return Response({'clients_amount': clients_amount})


@api_view(['GET'])
def current_client(request):
    return Response({"current_client": current_client})
