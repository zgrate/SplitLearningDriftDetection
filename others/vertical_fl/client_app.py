from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from sklearn.preprocessing import StandardScaler
import torch

from vertical_fl.task import ClientModel, load_data


class FlowerClient(NumPyClient):
    def __init__(self, v_split_id, data, lr):
        self.v_split_id = v_split_id
        self.data = torch.tensor(StandardScaler().fit_transform(data)).float()
        self.model = ClientModel(input_size=self.data.shape[1])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def get_parameters(self, config):
        return [val.cpu().detach().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param)
    def fit(self, parameters, config):
        # Update client-side model parameters
        self.set_parameters(parameters)

        # Forward pass on the client-side model
        intermediate_output = self.model(self.data)

        # Receive all gradients from the server via `config`
        all_gradients = config.get("server_gradients")
        client_index = config.get("client_index")

        if all_gradients is not None and client_index is not None:
            # Use the client-specific gradient
            server_gradients = torch.tensor(all_gradients[client_index])

            # Backward pass using the gradients from the server
            intermediate_output.backward(server_gradients)
            self.optimizer.step()

        # Return updated model parameters and send intermediate_output to server
        return (
            self.get_parameters(),
            len(self.train_data),
            {"intermediate_output": intermediate_output.detach().numpy()},
        )


    def evaluate(self, parameters, config):
        self.model.zero_grad()
        embedding = self.model(self.data)
        embedding.backward(torch.from_numpy(parameters[int(self.v_split_id)]))
        self.optimizer.step()
        return 0.0, 1, {}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    partition, v_split_id = load_data(partition_id, num_partitions=num_partitions)
    lr = context.run_config["learning-rate"]
    return FlowerClient(v_split_id, partition, lr).to_client()


app = ClientApp(
    client_fn=client_fn,
)
