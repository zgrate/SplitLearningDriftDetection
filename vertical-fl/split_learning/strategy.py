from typing import List, Tuple, Union, Optional, Dict

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, EvaluateRes, Scalar, FitRes, Parameters
from flwr.server.client_proxy import ClientProxy

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



# FL Allows reinitialization of local model each time
# Split Learning supposed we have a persistance on a local level
# This means that we need to store all data of the client on a server for local redistribution
# Idk if it is worth it
#



class Strategy(fl.server.strategy.FedAvg):
    def __init__(self, labels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = ServerModel(CLIENT_OUTPUT_INPUTS)
        self.initial_parameters = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.label = labels.clone().detach().float().unsqueeze(1)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print("Aggregate Fit")
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # print(results)metrics_ametrics_aggregatedggregated

        # optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        np_grads = []

        results_params = []
        i = 0
        for client, evaluate_results in results:
            # print(evaluate_results)
            print(i)
            i+=1
            params = parameters_to_ndarrays(evaluate_results.parameters)

            labels = torch.from_numpy(params[1])

            embeddings_aggregated = torch.from_numpy(params[0])
            print(labels)
            print(embeddings_aggregated)
            # embedding_server = torch.cat(embeddings_aggregated.detach().requires_grad_())

            # local_output = torch.cat(self.model.parameters(True))
            self.optimizer.zero_grad()
            embeding_server = self.model.requires_grad_()

            output = self.model(embeddings_aggregated)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()

            # output.backward()

            # grads = output.grad.split([4, 4, 4], dim=1)

            print(output)

            gradients = embeding_server.grad

            results_params.append(gradients.detach().numpy())

            # print(self.model[0].weight.grad)

            print("END")
            # gradients_ = output.grad.data.cpu().numpy()
            # loss = self.criterion(output, self.label)
            # loss.backward()

            # print(loss)
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

        return ndarrays_to_parameters(results_params), {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        print("START AGGREGATE EVALUATION")

        for client, evaluate_results in results:
            print(evaluate_results)

        print("THE END OF AGGREGATE EVALUATION")

        return None, {}
