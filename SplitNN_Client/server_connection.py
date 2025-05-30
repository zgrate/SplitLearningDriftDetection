import requests
import torch
from torch import Tensor

IP_ADDRESS = "http://localhost:8000"
TRAIN_API = "/train/"
TEST_API = "/test/"
PREDICT_API = "/predict/"
MASS_PREDICT_API = "/mass_predict/"
RESET_API = "/reset_runner/"
REPORT_NN_RESET = "/report_client_nn_reset/"
PREPARE_API = "/prepare_running/"
CURRENT_CLIENT_API = "/current_client/"
SAVE_REPORT_API = "/save_reports/"


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
            print("ERROR", torch.tensor(data['output']).isnan().any(), torch.tensor(data['output']).isinf().any())
            return None

    def train_request(self, intermid_output: Tensor, labels: Tensor, local_epoch: int = 0, last_comm_time: float = 0,
                      last_whole_training_time=0):
        response = self.post(TRAIN_API,
                             {"output": intermid_output.tolist(), "labels": labels.tolist(), "local_epoch": local_epoch,
                              "client_id": str(self.client_token), "last_comm_time": last_comm_time,
                              "last_whole_training_time": last_whole_training_time})
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

    def mass_prediction_request(self, input_data):
        response = self.post(MASS_PREDICT_API, {"input_data": input_data, "client_id": str(self.client_token), "local_epoch": None})
        if response.status_code == 200:
            return response.json()
        else:
            return None
