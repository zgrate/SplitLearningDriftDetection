import threading

from rest_framework.decorators import api_view
from rest_framework.response import Response

from data_logger.models import DataTransferLog, TrainingLog
from neural_network.models import global_server_model

import sys


def report_usage(method, data, client_id, direction_to_server=True):
    DataTransferLog(source_method=method, data_transfer_len=sys.getsizeof(data), target_source_client=client_id,
                    direction_to_server=direction_to_server).save()


def report_training(loss, client_id, epoch, server_epoch, mode="training"):
    TrainingLog(loss=loss, client_id=client_id, epoch=epoch, server_epoch=server_epoch, mode=mode).save()


lock = threading.Lock()

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

    local_epoch = request.data["local_epoch"]

    report_usage("train", request.data, client_id, direction_to_server=True)

    with lock:
        return_data, loss = global_server_model.train_input(request.data['output'], request.data['labels'])

        data = {"gradients": return_data, "loss": loss}
        # if return_data.isnan().any() or return_data.isinf().any():
        #     print("THIS SHOULD NOT BE ERROR!")
        #     return Response({})
        report_training(loss, client_id, local_epoch, global_server_model.epoch)

        report_usage("train", data, client_id, direction_to_server=False)
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
    local_epochs = request.data["local_epochs"]
    TrainingLog(mode="reset", client_id=client_id, epoch=local_epochs, server_epoch=global_server_model.epoch).save()


@api_view(["POST"])
def restart_runner(request):
    TrainingLog.objects.all().delete()
    DataTransferLog.objects.all().delete()
    global_server_model.reset_local_nn()
    return Response()
