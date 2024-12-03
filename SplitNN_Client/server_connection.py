import requests
import torch
from torch import Tensor

IP_ADDRESS = "http://localhost:8000"
TRAIN_API = "/train/"
TEST_API = "/test/"
PREDICT_API = "/predict/"
RESET_API = "/reset_runner/"
REPORT_NN_RESET = "/report_client_nn_reset/"

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

    def train_request(self, intermid_output: Tensor, labels: Tensor, local_epoch: int = 0):
        response = self.post(TRAIN_API, {"output": intermid_output.tolist(), "labels": labels.tolist(), "local_epoch": local_epoch, "client_id": str(self.client_token)})
        if response.status_code == 200:
            return response.json()

        return None

    def test_request(self, output: Tensor, test_labels: Tensor):
        response = self.post(TEST_API, {"output": output.tolist(), "labels": test_labels.tolist(), "client_id": str(self.client_token)})
        if response.status_code == 200:
            return response.json()

        return None

    def predict_request(self, output):
        response = self.post(PREDICT_API, {"output": output.tolist(), "client_id": str(self.client_token)})
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