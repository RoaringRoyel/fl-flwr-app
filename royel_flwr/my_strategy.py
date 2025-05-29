from flwr.server.strategy import FedAvg
from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
import torch
from .task import Net, set_weights

class CustomFedAvg(FedAvg):
    """Custom FedAvg strategy with additional functionality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate_fit(self,
                    server_round:int,
                    results: list[tuple[ClientProxy, FitRes]],
                    failures: list[tuple[ClientProxy, FitRes] | BaseException]
                    ) -> tuple[Parameters | None, dict[str, bool| bytes | float |int |str]]:
                    
            parameteres_aggragated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

            ndarrays = parameters_to_ndarrays(parameteres_aggragated)
            model = Net()
            set_weights(model, ndarrays)

            torch.save(model.state_dict(), f"Global_model_round_{server_round}")

            return parameteres_aggragated, metrics_aggregated
