import random

from torch import Tensor


import torch

class AbstractDrifter:

    def __init__(self, *args, **kwargs):
        pass

    def activate_drifting(self, iterator):
        return iterator

    def next_drifting(self, next_elements: list[list[Tensor], list[Tensor]]):
        return next_elements


class RandomDrifter(AbstractDrifter):

    def __init__(self, drifting_probability, seed):
        self.drifting_probability = drifting_probability
        self.seed = seed
        self.random = random.Random(seed)


    def next_drifting(self,  next_elements):
        X: Tensor
        y: Tensor
        (X, y) = next_elements

        def drift(value):
            if self.random.random() < self.drifting_probability:
                return value*self.random.random()
            else:
                return value


        X.apply_(drift)
#
# if __name__ == '__main__':
#     from data_provider import MNISTDataInputStream, get_test_training_data, DataInputDataset, \
#         DriftDatasetLoader
#
#     data_input_stream = MNISTDataInputStream(*get_test_training_data(0, 4))
#     dataset = DataInputDataset(data_input_stream)
#     training_data, validation_data = torch.utils.data.random_split(dataset, [0.9, 0.1])
#     # training_data_loader = torch.utils.data.DataLoader(training_data,
#     #                                     batch_size=len(training_data.indices))
#     validation_data_loader = DriftDatasetLoader(validation_data, RandomDrifter(0.1, 1),
#                                           batch_size=len(validation_data.indices))
#     validation_data_loader.active = True
#     for X, Y in validation_data_loader:
#         print(X)
#         print(Y)