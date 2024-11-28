import requests
from torch import Tensor

IP_ADDRESS = "http://localhost:8000"
TRAIN_API = "/train/"
TEST_API = "/test/"
PREDICT_API = "/predict/"

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
        return self.session.post(url=IP_ADDRESS + url, json=data)

    def train_request(self, intermid_output: Tensor, labels: Tensor):
        response = self.post(TRAIN_API, {"output": intermid_output.tolist(), "labels": labels.tolist()})
        if response.status_code == 200:
            return response.json()

        return None

    def test_request(self, output: Tensor, test_labels: Tensor):
        print(output)
        response = self.post(TEST_API, {"output": output.tolist(), "labels": test_labels.tolist()})
        if response.status_code == 200:
            return response.json()

        return None

    def predict_request(self, output):
        response = self.post(PREDICT_API, {"output": output.tolist()})
        if response.status_code == 200:
            return response.json()

        return None