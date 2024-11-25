import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from split_learning.globals import CLIENT_OUTPUT_INPUTS



# models = [
#     nn.Sequential(
#         #ClientLayer(),
#         nn.Flatten(),
#         nn.Linear(input_size, hidden_sizes[0]),
#         nn.ReLU(),
#         nn.Linear(hidden_sizes[0], hidden_sizes[1]),
#         nn.ReLU(),
#     ),
#     nn.Sequential(nn.Linear(hidden_sizes[1], output_size), nn.LogSoftmax(dim=1)),
# ]
class ServerModel(nn.Module):
    def __init__(self, input_size):
        super(ServerModel, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, out_features=10), nn.LogSoftmax(dim=1))

    def forward(self, x):
        return self.model(x)



class Strategy(fl.server.strategy.FedAvg):
    def __init__(self, labels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = ServerModel(CLIENT_OUTPUT_INPUTS)
        self.initial_parameters = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        )
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.BCELoss()
        self.label = torch.tensor(labels).float().unsqueeze(1)

    def aggregate_fit(
        self,
        rnd,
        results,
        failures,
    ):
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # print(results)metrics_ametrics_aggregatedggregated

        np_grads = []

        for  _, fit_res in results:
            embeddings_aggregated = torch.from_numpy(parameters_to_ndarrays(fit_res.parameters)[0])
            embedding_server = torch.cat(embeddings_aggregated.detach().requires_grad_())

            local_output = torch.cat(self.model.parameters(True))
            output = self.model(embedding_server)


            gradients_ = output.grad.data.cpu().numpy()


            loss = self.criterion(output, self.label)
            loss.backward()
            # self.optimizer.step()
            # self.optimizer.zero_grad()
            # print(output)


        # # Convert results
        # embedding_results = [
        #     torch.from_numpy(parameters_to_ndarrays(fit_res.parameters)[0])
        #     for _, fit_res in results
        # ]

        # print(embedding_results)

        # embeddings_aggregated = torch.cat(embedding_results, dim=1)
        # embedding_server = embeddings_aggregated.detach().requires_grad_()
        # output = self.model(embedding_server)
        # loss = self.criterion(output, self.label)
        # loss.backward()
        #
        # self.optimizer.step()
        # self.optimizer.zero_grad()

        # grads = embedding_server.grad.split([4, 4, 4], dim=1)
        # np_grads = [grad.numpy() for grad in grads]
        # parameters_aggregated = ndarrays_to_parameters(np_grads)utput = self.model(embedding_server)
        #
        # with torch.no_grad():
        #     correct = 0
        #     output = self.model(embedding_server)
        #     predicted = (output > 0.5).float()
        #
        #     correct += (predicted == self.label).sum().item()
        #
        #     accuracy = correct / len(self.label) * 100
        #
        # metrics_aggregated = {"accuracy": accuracy}

        return 0, 0

    def aggregate_evaluate(
        self,
        rnd,
        results,
        failures,
    ):
        return None, {}
