from time import sleep

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from sklearn.preprocessing import StandardScaler
import torch
from torchvision.datasets import MNIST
from torchvision.transforms.v2 import ToTensor

from split_learning.task import ClientModel, load_data


class FlowerClient(NumPyClient):
    def __init__(self, v_split_id, data, labels, lr):
        self.v_split_id = v_split_id
        self.data = data.float()
        self.labels = labels
        assert len (data) == len (labels)
        self.model = ClientModel(input_size=self.data.shape[1])

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    # Return the current local model parameters
    def get_parameters(self, config):
        print("GET PARAMENETS ")
        pass

    # Receive model parameters from the server, train the model on the local data, and return the updated model parameters to the server
    def fit(self, parameters_from_server, config):
        print("FIT PARAMENTERS")
        embedding = self.model(self.data)
        emb = (embedding.detach().numpy())
        return [emb, self.labels], len(emb), {}

    # Receive model parameters from the server, evaluate the model on the local data, and return the evaluation result to the server
    def evaluate(self, parameters, config):
        print("BACK EVALUATE")
        print(int(self.v_split_id))
        print(parameters[int(self.v_split_id)])
        print("END BACK EVALUATE")
        cel = torch.nn.CrossEntropyLoss()
        embedding = self.model(self.data)

        loss = cel(embedding, torch.from_numpy(parameters[int(self.v_split_id)]))
        loss.backward()
        # embedding.backward()
        self.optimizer.step()
        return float(loss), 1, {}


def client_fn(context: Context):

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])


    mnist = MNIST("mnists/", download=True)
    le = len(mnist.data)/num_partitions

    print("Client Starting,,,", partition_id, num_partitions, le, partition_id*le, (partition_id+1)*le)
    # partition, v_split_id = load_data(partition_id, num_partitions=num_partitions)
    # lr = context.run_config["learning-rate"]
    lr = 0.1
    start = int(partition_id*le)
    end = int((partition_id+1)*le)
    return FlowerClient(partition_id, mnist.data[start:end], mnist.targets[start:end], lr).to_client()


client = ClientApp(
    client_fn=client_fn,
)
