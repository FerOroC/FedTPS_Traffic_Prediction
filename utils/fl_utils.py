import torch
import numpy as np
import scipy.sparse as sp
import dgl
import torch.nn as nn 
import pandas as pd 
import os
import flwr as fl
from typing import List, Tuple, Union, Optional, Dict, Callable
from collections import OrderedDict
from pathlib import Path
import pickle

import argparse 

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

# set and get model parameters via flower functions
def set_parameters(model, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

# # server centric evaluation method
# def get_evaluate_fn(model, test_iter, epochs, metrics, data_mean, data_std):

#     def evaluate(
#         server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
#     ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
#         set_parameters(model, parameters)
#         MAE, RMSE, MAPE, loss = test(model, test_iter, data_mean, data_std)
#         print(loss)
#         metrics["train_losses"][0].append(loss)
#         print(server_round)
#         print(epochs)
#         if (server_round%epochs == 0) and server_round>0:
#             print("AAAAAAAAAAA")
#             set_parameters(model, parameters)  # Update model with the latest parameters
#             MAE, RMSE, MAPE, loss = test(model, test_iter, data_mean, data_std)
#             metrics['test_mae'].append(MAE)
#             metrics['test_rmse'].append(RMSE)
#             metrics['test_mape'].append(MAPE)
#             return loss, {"MAPE": MAPE}

    # return evaluate


def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

# implement Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, testloader, data_mean, data_std, local_rounds, lr):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.data_mean = data_mean
        self.data_std = data_std
        self.testloader = testloader
        self.local_rounds = local_rounds
        self.lr = lr

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        # if FedProx being used, and proximal mu provided by strategy, then run FedProx specific training with clipping
        if config:
            if config["proximal_mu"]:
                train_prox(self.model, self.trainloader, local_epochs=self.local_rounds, mu=config["proximal_mu"], lr=self.lr)
        else:
            train(self.model, self.trainloader, local_epochs=self.local_rounds, lr=self.lr)
        return get_parameters(self.model), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        MAE, RMSE, MAPE, loss = test(self.model, self.testloader, self.data_mean, self.data_std)
        #print("MAE: ", MAE, ", RMSE: ", RMSE, ", MAPE: ", MAPE, ", Loss: ", loss)
        return float(loss), len(self.valloader), {"MAPE": float(MAPE), "MAE": float(MAE), "RMSE": float(RMSE)}


class FedBNFlowerClient(FlowerClient):
    """Similar to FlowerClient but this is used by FedBN clients."""

    def __init__(self, save_path: Path, client_id: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # For FedBN clients we need to persist the state of the BN
        # layers across rounds. In Simulation clients are statess
        # so everything not communicated to the server (as it is the
        # case as with params in BN layers of FedBN clients) is lost
        # once a client completes its training. An upcoming version of
        # Flower suports stateful clients
        save_path = Path(save_path)
        bn_state_dir = save_path / "bn_states"
        bn_state_dir.mkdir(exist_ok=True)
        self.bn_state_pkl = bn_state_dir / f"client_{client_id}.pkl"

    def _save_bn_statedict(self) -> None:
        """Save contents of state_dict related to BN layers."""
        bn_state = {
            name: val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if "bn" in name
        }

        with open(self.bn_state_pkl, "wb") as handle:
            pickle.dump(bn_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_bn_statedict(self) -> Dict[str, torch.tensor]:
        """Load pickle with BN state_dict and return as dict."""
        with open(self.bn_state_pkl, "rb") as handle:
            data = pickle.load(handle)
        bn_stae_dict = {k: torch.tensor(v) for k, v in data.items()}
        return bn_stae_dict

    def get_parameters(self, config) -> NDArrays:
        """Return model parameters as a list of NumPy ndarrays w or w/o using BN.

        layers.
        """
        # First update bn_state_dir
        self._save_bn_statedict()
        # Excluding parameters of BN layers when using FedBN
        return [
            val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if "bn" not in name
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy ndarrays Exclude the bn layer if.

        available.
        """
        keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)

        # Now also load from bn_state_dir
        if self.bn_state_pkl.exists():  # It won't exist in the first round
            bn_state_dict = self._load_bn_statedict()
            self.model.load_state_dict(bn_state_dict, strict=False)

class FedAvgSave(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        checkpoint_path: str, 
        n_pred: int,
        epochs: int, 
        model,
        test_iter,
        data_std,
        data_mean,
        metric_results,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.n_pred = n_pred
        self.epochs = epochs
        self.model = model
        self.test_iter = test_iter
        self.data_std = data_std
        self.data_mean = data_mean
        self.metric_results = metric_results
        self.initial_parameters = initial_parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            if int(server_round)%20==0:
                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                # Save aggregated_ndarrays
                print(f"Saving round {server_round} aggregated_ndarrays...")
                round_save_path = self.checkpoint_path + f"/epoch{server_round}_model_{self.n_pred}pred.npz"
                np.savez(round_save_path, *aggregated_ndarrays)

            if (int(server_round)%self.epochs==0) and (server_round>0):
                params = parameters_to_ndarrays(aggregated_parameters)
                set_parameters(self.model, params)
                MAE, RMSE, MAPE, loss = test(self.model, self.test_iter, self.data_mean, self.data_std)
                self.metric_results['test_mae'].append(MAE)
                self.metric_results['test_rmse'].append(RMSE)
                self.metric_results['test_mape'].append(MAPE)

            if (server_round%10==0) or (server_round==1):
                params = parameters_to_ndarrays(aggregated_parameters)
                set_parameters(self.model, params)
                MAE, RMSE, MAPE, loss = test(self.model, self.test_iter, self.data_mean, self.data_std)
                self.metric_results['eval_metrics'][0].append(MAE)
                print("eval metrics: ", MAE)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        # Call aggregate_evaluate from base class to evaluate aggregated parameters
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        self.metric_results['train_losses'][0].append(loss)

        return loss, metrics

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

class FedProxSave(fl.server.strategy.FedProx):
    def __init__(
        self,
        *,
        checkpoint_path: str, 
        n_pred: int,
        epochs: int, 
        model,
        test_iter,
        data_std,
        data_mean,
        metric_results,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        proximal_mu: float = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
        super().__init__(
            proximal_mu = proximal_mu
        )
        self.checkpoint_path = checkpoint_path
        self.n_pred = n_pred
        self.epochs = epochs
        self.model = model
        self.test_iter = test_iter
        self.data_std = data_std
        self.data_mean = data_mean
        self.metric_results = metric_results
        self.initial_parameters = initial_parameters
        

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            if int(server_round)%20==0:
                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                # Save aggregated_ndarrays
                print(f"Saving round {server_round} aggregated_ndarrays...")
                round_save_path = self.checkpoint_path + f"/epoch{server_round}_model_{self.n_pred}pred.npz"
                np.savez(round_save_path, *aggregated_ndarrays)

            if (int(server_round)%self.epochs==0) and (server_round>0):
                params = parameters_to_ndarrays(aggregated_parameters)
                set_parameters(self.model, params)
                MAE, RMSE, MAPE, loss = test(self.model, self.test_iter, self.data_mean, self.data_std)
                self.metric_results['test_mae'].append(MAE)
                self.metric_results['test_rmse'].append(RMSE)
                self.metric_results['test_mape'].append(MAPE)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        # Call aggregate_evaluate from base class to evaluate aggregated parameters
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        self.metric_results['train_losses'][0].append(loss)

        return loss, metrics

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters


class FedOptSave(fl.server.strategy.FedOpt):
    def __init__(
        self,
        *,
        checkpoint_path: str, 
        n_pred: int,
        epochs: int, 
        model,
        test_iter,
        data_std,
        data_mean,
        metric_results,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        beta_1: float = 0.0,
        beta_2: float = 0.0,
        tau: float = 1e-9,
    ) -> None:
        super().__init__(
            initial_parameters = initial_parameters,
            eta = eta,
            eta_l = eta_l,
            tau = tau,
            beta_1 = beta_1,
            beta_2 = beta_2
        )
        self.checkpoint_path = checkpoint_path
        self.n_pred = n_pred
        self.epochs = epochs
        self.model = model
        self.test_iter = test_iter
        self.data_std = data_std
        self.data_mean = data_mean
        self.metric_results = metric_results
        self.initial_parameters = initial_parameters
        

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            if int(server_round)%20==0:
                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                # Save aggregated_ndarrays
                print(f"Saving round {server_round} aggregated_ndarrays...")
                round_save_path = self.checkpoint_path + f"/epoch{server_round}_model_{self.n_pred}pred.npz"
                np.savez(round_save_path, *aggregated_ndarrays)

            if (int(server_round)%self.epochs==0) and (server_round>0):
                params = parameters_to_ndarrays(aggregated_parameters)
                set_parameters(self.model, params)
                MAE, RMSE, MAPE, loss = test(self.model, self.test_iter, self.data_mean, self.data_std)
                self.metric_results['test_mae'].append(MAE)
                self.metric_results['test_rmse'].append(RMSE)
                self.metric_results['test_mape'].append(MAPE)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        # Call aggregate_evaluate from base class to evaluate aggregated parameters
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        self.metric_results['train_losses'][0].append(loss)

        return loss, metrics

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters


# train function
def train(model, train_iter, local_epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_criteria = nn.MSELoss()
    for epoch in range(1, local_epochs+1):
        l_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            y_pred = model(x)
            y_pred = torch.squeeze(y_pred)
            l = loss_criteria(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        print("Epoch: ", epoch, ", Train loss:", l_sum / n)

def train_prox(model, train_iter, local_epochs, mu, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_criteria = nn.MSELoss()
    global_params = [val.detach().clone() for val in model.parameters()]
    for epoch in range(1, local_epochs+1):
        l_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            y_pred = model(x)
            y_pred = torch.squeeze(y_pred)
            optimizer.zero_grad()
            proximal_term = 0.0

            for local_weights, global_weights in zip(model.parameters(), global_params):
                proximal_term += torch.square((local_weights - global_weights).norm(2))

            l = loss_criteria(y_pred, y) + (mu/2) * proximal_term

            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        print("Epoch: ", epoch, ", Train loss:", l_sum / n)





# test function

#we want to pass test_iter which has distinct data_iters, one for each client 
# we want to evaluate MAE and RMSE at each point, before we mean them, add all the values for a given region between all clients 
# then we do the mean
# so we no longer normalise the metrics.
def test(model, test_iter, mean, std):
    model.eval()
    criterion = nn.MSELoss()
    mean = mean.mean()
    std = std.mean()
    with torch.no_grad():
        l_sum, n = 0.0, 0
        mape = []
        mape_homog = []
        cum_mae, cum_mse = [], []
        for data_iter in test_iter:
            mae, mse = [],[]
            for x, y in data_iter:
                y_pred = model(x)
                y_pred = torch.squeeze(y_pred)
                l = criterion(y_pred, y)
                l_sum += l.item() * y.shape[0]
                n += y.shape[0]
                y = y.detach().cpu().numpy()
                y = (y * std) + mean                    #unnormalised y
                y_pred = y_pred.detach().cpu().numpy()
                y_pred = (y_pred * std) + mean          # unnormalised y_pred
                d = np.abs(y - y_pred)
                mae += d.tolist()
                mape.append(evaluate_mape(y,y_pred))
                mse += (d**2).tolist()
            cum_mae.append(mae)
            cum_mse.append(mse)

        total_mae = []
        total_mse = []

        for i in range(len(test_iter[0])):
            sum_value = sum(sublist[i] for sublist in mae)
            total_mae.append(sum_value)
            sum_value = sum(sublist[i] for sublist in mse)
            total_mse.append(sum_value)

        MAE = np.array(mae).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        for item in mape:
            mape_homog.append(item.mean())
        MAPE = np.array(mape_homog).mean()
        return MAE, RMSE, MAPE, l_sum/n

def evaluate_mape(y, y_pred, threshold=2):
    y = y
    y_pred = y_pred
    mask = y > threshold
    if np.sum(mask)!=0:
        mape = np.abs(y[mask] - y_pred[mask])/y[mask]
        return mape
    else:
        return np.nan