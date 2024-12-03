from torch import nn


class ServerModelWrapper(nn.Module):
    def __init__(self):
        super(ServerModelWrapper, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, out_features=50),
            nn.ReLU(),
            nn.Linear(50, out_features=10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)