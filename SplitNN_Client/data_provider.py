from dataclasses import dataclass
from typing import Dict

import numpy
import torch.utils.data
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torchvision import transforms
from torchvision.datasets import MNIST

from SplitNN_Client.drifting_simulation import AbstractDrifter


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


    # print(data)

    le = len(mnist.train_data) / client_count

    start = int(client_id * le)
    end = int((client_id + 1) * le)



    part_data = mnist.train_data[start:end].float()/255
    part_targets = mnist.train_labels[start:end].long()

    return part_data, part_targets, [], []


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

    def start_drifting(self):
        pass

    def __init__(self, train, labels, test, labels_test):
        super().__init__()
        self.data = DataChunk(train, labels, test, labels_test)

    def get_data_part(self):
        return self.data


class DataInputDataset(Dataset):
    def __init__(self, data_input_stream: AbstractDataInputStream, train=True):
        super().__init__()
        self.data_input_stream = data_input_stream
        self.train = train


    def __getitem__(self, idx):
        if self.train:
            return self.data_input_stream.get_data_part().train_data[idx], self.data_input_stream.get_data_part().train_labels[idx]
        else:
            return self.data_input_stream.get_data_part().test_data[idx], self.data_input_stream.get_data_part().test_labels[idx]

    def __len__(self):
        return len(self.data_input_stream.get_data_part().train_data if self.train else self.data_input_stream.get_data_part().test_data)


class DriftDatasetLoader(DataLoader):

    iterator: _BaseDataLoaderIter

    def __init__(self, dataset: Dataset, drifter: AbstractDrifter, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.drifter = drifter
        self.active = False

    def __iter__(self):
        self.iterator = super().__iter__()
        #TODO: Here we need to init the drifter
        return self

    def __next__(self):
        next_items = next(self.iterator)
        return self.drifter.next_drifting(next_items) if self.active else next_items
