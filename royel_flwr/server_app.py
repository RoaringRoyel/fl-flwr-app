"""royel-flwr: A Flower / PyTorch app."""

from typing import List, Tuple #for weghted_average callback function
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from royel_flwr.task import Net, get_weights

def weighted_average(metrics: List[tuple[int, Metrics]]) -> Metrics:
    """ A function that aggregates metrics from clients. """
    accuracies = [num_examples * m["accuracy"] for num_examples,m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total_examples}



def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=8,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,  # Custom aggregation function
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)