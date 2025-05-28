"""royel-flwr: A Flower / PyTorch app."""

import torch
from random import random
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from royel_flwr.task import Net, get_weights, load_data, set_weights, test, train
import json # For complex metrics serialization

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        print(config)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            config["lr"], # learning rate from config will got to the TASK.py
            self.device,
        )

        #creating a complex matrix of metrics
        complex_metrics = {"a": 1234, "b": 5423, "Royel": random()}
        complex_metric_str= json.dumps(complex_metrics)  # Convert to JSON string if needed
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss ,"my_metric" : complex_metric_str}  # Example of adding a random number to metrics,
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
