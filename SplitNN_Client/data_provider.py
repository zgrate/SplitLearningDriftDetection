from dataclasses import dataclass
from typing import Dict

from torch import Tensor
from torchvision.datasets import MNIST


# def client_fn(context: Context):
#     partition_id = int(context.node_config["partition-id"])
#     num_partitions = int(context.node_config["num-partitions"])
#
#     mnist = MNIST("mnists/", download=True)
#     le = len(mnist.data) / num_partitions
#
#     print("Client Starting,,,", partition_id, num_partitions, le, partition_id * le, (partition_id + 1) * le)
#     # partition, v_split_id = load_data(partition_id, num_partitions=num_partitions)
#     # lr = context.run_config["learning-rate"]
#     lr = 0.1
#     start = int(partition_id * le)
#     end = int((partition_id + 1) * le)
#     return FlowerClient(partition_id, mnist.data[start:end], mnist.targets[start:end], lr).to_client()


def division_data(clients_number):
    mnist = MNIST("mnists/", download=True)

    return [get_test_training_data(i, clients_number, mnist) for i in range(clients_number)]


def get_test_training_data(client_id, client_count, mnist=None):
    if mnist is None:
        mnist = MNIST("mnists/", download=True)

    le = len(mnist.data) / client_count

    start = int(client_id * le)
    end = int((client_id + 1) * le)

    return mnist.data[start:end].float(), mnist.targets[start:end].float(), mnist.test_data.float(), mnist.test_labels.float()


@dataclass
class DataChunk:
    train_data: Tensor
    train_labels: Tensor
    test_data: Tensor
    test_labels: Tensor


class AbstractDataInputStream:

    def get_data_part(self) -> DataChunk:
        raise NotImplementedError()


class MNISTDataInputStream(AbstractDataInputStream):

    def __init__(self, train, labels, test, labels_test):
        super().__init__()
        self.data = DataChunk(train, labels, test, labels_test)

    def get_data_part(self):
        return self.data