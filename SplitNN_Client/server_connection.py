import os
import time

import requests
import torch
from torch import Tensor

IP_ADDRESS = "http://localhost:8000"
TRAIN_API = "/train/"
TEST_API = "/test/"
PREDICT_API = "/predict/"
RESET_API = "/reset_runner/"
REPORT_NN_RESET = "/report_client_nn_reset/"
PREPARE_API = "/prepare_running/"
CURRENT_CLIENT_API = "/current_client/"
SAVE_REPORT_API = "/save_reports/"
CLIENT_CHECK_API = "/client_check/"


class ServerConnection:
    def __init__(self, client_token):
        self.client_token = client_token

    @property
    def session(self):
        session = requests.Session()
        return session

    def get(self, address):
        return self.session.get(IP_ADDRESS + address)

    def put(self, url, data):
        return self.session.put(url=IP_ADDRESS + url, json=data)

    def post(self, url, data):
        try:
            return self.session.post(url=IP_ADDRESS + url, json=data)
        except ValueError:
            # print("ERROR", torch.tensor(data['output']).isnan().any(), torch.tensor(data['output']).isinf().any())
            return None

    def check_request(self, request_id):
        response = self.post(CLIENT_CHECK_API, {
            "client_id": str(self.client_token),
            "request_id": str(request_id)
        })
        print(response.status_code)
        if response.status_code == 204:
            return False
        elif response.status_code == 200:
            return response.json()

        return None


    def train_request(self, intermid_output: Tensor, labels: Tensor, local_epoch: int = 0, last_comm_time: float = 0,
                      last_whole_training_time=0, gpu_runner=False, retry=120):
        response = self.post(TRAIN_API,
                             {"output": intermid_output.tolist(), "labels": labels.tolist(), "local_epoch": local_epoch,
                              "client_id": str(self.client_token), "last_comm_time": last_comm_time,
                              "last_whole_training_time": last_whole_training_time})

        if gpu_runner:
            if response.status_code == 200:
                requst_id = response.json()['request_id']
                print("Going into waiting loop...")
                while retry > 0:
                    response = self.check_request(requst_id)
                    if response is None:
                        return None

                    elif response == False:
                        retry -= 1
                    else:
                        return response
                    if retry % 10 == 0:
                        print(f"Retrying {retry} times")

                    time.sleep(0.5)

                return None
        if response.status_code == 200:
            return response.json()

        return None

    def test_request(self, output: Tensor, test_labels: Tensor, local_epoch, type):
        response = self.post(TEST_API, {"output": output.tolist(), "labels": test_labels.tolist(),
                                        "client_id": str(self.client_token), "local_epoch": local_epoch, "type": type})
        if response.status_code == 200:
            return response.json()

        return None

    def predict_request(self, output, local_epoch=None, target_label=None):
        response = self.post(PREDICT_API, {"output": output.tolist(), "client_id": str(self.client_token), "target_label": target_label, "local_epoch": local_epoch})
        if response.status_code == 200:
            return response.json()

        return None

    def reset_runner(self):
        response = self.post(RESET_API, {})
        if response.status_code == 200:
            return True

        return False

    def report_reset_nn(self):
        response = self.post(REPORT_NN_RESET, {"client_id": self.client_token})
        if response.status_code == 200:
            return True

        return False

    def save_report(self, details):
        response = self.post(SAVE_REPORT_API, {"details": {"client_id": self.client_token, **details}})
        if response.status_code == 200:
            return response.json()

        return False

    def prepare_runner(self, all_props):
        response = self.post(PREPARE_API, all_props)
        print(response.content)
        if response.status_code == 200:
            return True

        return False

    def current_client(self):
        response = self.get(CURRENT_CLIENT_API)
        if response.status_code == 200:
            return response.json()['current_client']
        else:
            return -1
