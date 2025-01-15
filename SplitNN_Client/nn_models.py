from torch import nn, Tensor


class ClientServerModel0:

    client = lambda : nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
    )

    server = lambda :nn.Sequential(
            nn.Linear(128, 100),
            nn.ReLU(),
            nn.Linear(100, out_features=128),
            nn.ReLU(),
            nn.Linear(128, out_features=100),
            nn.ReLU(),
            nn.Linear(100, out_features=10),
        )


class ClientServerModel1:

    client = lambda :nn.Sequential(
        # ClientLayer(),
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 100),
        nn.ReLU(),
    )

    server = lambda :nn.Sequential(
            nn.Linear(100, out_features=128),
            nn.ReLU(),
            nn.Linear(128, out_features=100),
            nn.ReLU(),
            nn.Linear(100, out_features=10),
        )



class ClientServerModel2:

    client = lambda :nn.Sequential(
        # ClientLayer(),
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 100),
        nn.ReLU(),
        nn.Linear(100, out_features=128),
        nn.ReLU(),
    )

    server = lambda :nn.Sequential(
            nn.Linear(128, out_features=100),
            nn.ReLU(),
            nn.Linear(100, out_features=10),
        )

class ClientServerModel3:
    client = lambda : nn.Sequential(
        # ClientLayer(),
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 100),
        nn.ReLU(),
        nn.Linear(100, out_features=128),
        nn.ReLU(),
        nn.Linear(128, out_features=100),
        nn.ReLU(),
    )

    server = lambda : nn.Sequential(
        nn.Linear(100, out_features=10),
    )
