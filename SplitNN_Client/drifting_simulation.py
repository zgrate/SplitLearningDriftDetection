import random

from torch import Tensor


class AbstractDrifter:

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
        for X, y in next_elements:
            print(X)
            X.apply_(lambda d: d+1)
            print(X)
            exit(0)

