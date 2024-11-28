import threading

from rest_framework.decorators import api_view
from rest_framework.response import Response

from data_logger.models import DataTransferLog
from neural_network.models import global_server_model

import sys

def report_usage(method, data, client_id, direction_to_server=True):
    DataTransferLog(source_method=method, data_transfer_len=sys.getsizeof(data), target_source_client=client_id, direction_to_server=direction_to_server).save()


lock = threading.Lock()

# Create your views here.
@api_view(["POST"])
def train(request):
    report_usage("train", request.data, "no_client_yet", direction_to_server=True)

    with lock:
        return_data, loss = global_server_model.train_input(request.data['output'], request.data['labels'])

    data = {"gradients": return_data, "loss": loss}

    report_usage("train", data, "no_client_yet", direction_to_server=False)
    return Response(data)


@api_view(["POST"])
def test(request):
    report_usage("test", request.data, "no_client_yet", direction_to_server=True)

    with lock:
        loss = global_server_model.test(request.data['output'], request.data['labels'])

    report_usage("test", loss, "no_client_yet", direction_to_server=False)

    return Response({"loss": loss})


@api_view(["POST"])
def predict(request):
    report_usage("predict", request.data, "no_client_yet", direction_to_server=True)
    with lock:
        data = {"predicted": global_server_model.predict(request.data['output'])}

    report_usage("predict", data, "no_client_yet", direction_to_server=False)

    return Response(data)

