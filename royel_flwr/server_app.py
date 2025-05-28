"""royel-flwr: A Flower / PyTorch app."""

from typing import List, Tuple #for weghted_average callback function
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from royel_flwr.task import Net, get_weights, set_weights, test, get_transforms
from datasets import load_dataset
from torch.utils.data import DataLoader

def get_evaulate_fn(testloader, device):
    """Return a callback that evaluates the global model on the test set."""
    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model using provided centralised testset."""
        net = Net()
        # Set model parameters
        set_weights(net, parameters_ndarrays)
        net.to(device)
        # Evaluate model on test set
        loss, accuracy = test(net,testloader,device)

        return loss, {"centralized_accuracy": accuracy}
    return evaluate

#eval_metrics = [(res.num_examples, res.metrics) for _, res in results] [from Defination]
def weighted_average(metrics: List[tuple[int, Metrics]]) -> Metrics:
    """ A function that aggregates metrics from clients. """
    accuracies = [num_examples * m["accuracy"] for num_examples,m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total_examples}



def on_fit_config(server_round: int) -> Metrics:
    """Callback function to set the fit configuration for each round."""
    """Adjust learning rate or other parameters based on the round number."""
    lr = 0.01
    if server_round > 2:
        lr = 0.005
    return{"lr": lr}

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    #load a global test set
    testset = load_dataset("zalando-datasets/fashion_mnist")["test"]

    testloader = DataLoader(testset.with_transform(get_transforms()), batch_size=32)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=8,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,  # Custom aggregation function
        on_fit_config_fn=on_fit_config,  # Callback to set fit config
        evaluate_fn= get_evaulate_fn(testloader, device="cpu"),  # Function to evaluate model on test set
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)