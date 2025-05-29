from flwr.server.strategy import FedAvg
from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
import torch
from .task import Net, set_weights
import json
import wandb
from datetime import datetime
# Custom FedAvg strategy that extends the default FedAvg with additional functionality

class CustomFedAvg(FedAvg):
    """Custom FedAvg strategy with additional functionality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #Dictionary to store results for each round
        self.results_to_save = {} #for saving the result matrics in each round

        #Log those same metrics to W&B
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(project="flower-simulation-tutorial", name=f"custom-strategy-{name}")


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
    
    def evaluate(self,
                 server_round: int,
                 parameters: Parameters
                 )-> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
            loss, metrics = super().evaluate(server_round, parameters)

            my_results= {"loss": loss, **metrics} 

            self.results_to_save[server_round] = my_results

           #save as a JSON file
            with open("results_json",'w') as json_file:
                 json.dump(self.results_to_save, json_file, indent=4)

            # Log metrics to W&B
            wandb.log(my_results, step = server_round)
            return loss, metrics
