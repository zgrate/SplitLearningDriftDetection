from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation

from split_learning.client_app import client
from split_learning.strategy import Strategy
from split_learning.task import process_dataset


# For client in client
#     DATA -> Client NN -> Cut Layer results -> Transfer -> Server Input -> Server NN -> Pred Labels
#
# 1.  Pred Label + Whole Server NN -> Client -> Client + Server NN -> Evaluate -> Send back Server NN paramerts
#
#

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""

    # Get dataset
    processed_df = process_dataset()
    print(processed_df.train_labels)

    # Define the strategy
    strategy = Strategy(processed_df.train_labels.unique())

    # Construct ServerConfig
    # num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=10)

    return ServerAppComponents(strategy=strategy, config=config)


# Start Flower server
app = ServerApp(server_fn=server_fn)

backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

run_simulation(
    server_app=app,
    client_app=client,
    num_supernodes=2,
    backend_config=backend_config,
)