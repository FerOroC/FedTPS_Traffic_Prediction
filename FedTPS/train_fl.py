import json
# remove conditional info stuff argument into functino call and everything relating to this
import torch
import numpy as np
import scipy.sparse as sp
import dgl
import torch.nn as nn 
import pandas as pd 
import os
import flwr as fl
from typing import List, Tuple, Union, Optional, Dict
from collections import OrderedDict
from config import params

import argparse 

from models.gru import GRU
from models.dcrnn import DCRNN
from models.gwnet import GWNET
from models.stgcn import STGCN
from models.tau import SimVP
from models.gatau import GATAU
from csv import writer

from utils.utils import load_chengdu_data_new, data_loader, data_transform, evaluate_mape, evaluate_model, load_adj
from utils.fl_utils import set_parameters, get_parameters, FlowerClient, FedAvgSave, FedProxSave, FedOptSave
from types import SimpleNamespace



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

window = 12
n_pred = 6

batch_size = 32
n = 100

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

# make result and checkpoint directories to store results in 
parser = argparse.ArgumentParser()
parser.add_argument("--num_clients", required=True, help="Number of clients in FL setting", type=int)
parser.add_argument("--model_name", required=True, help="Name of model", type=str)
parser.add_argument("--epochs", required=True, help="Number of epochs to run for", type=int)
parser.add_argument("--city", required=True, help="Name of city", type=str)
parser.add_argument("--FL_method", required=True, help="FL method", type=str)
parser.add_argument("--gen_samples", required=True, help="Number of synthetic samples", type=int)
parser.add_argument("--loc_ratio", type=int, default=1)
parser.add_argument("--initial_params", type=bool, default=False)
args = vars(parser.parse_args())

num_clients = args['num_clients']
model_name = args['model_name']
epochs = args['epochs']
city = args['city']
FL_method = args['FL_method']
loc_ratio = args['loc_ratio']
gen_samples = args['gen_samples']
initial_params = bool(args['initial_params'])

print("num clients: ", num_clients)
print("model name: ", model_name)
print("FL method: ", FL_method)
# Load configuration
temp = {}
for k, v in params.items():
    temp[k] = SimpleNamespace(**v)
config = SimpleNamespace(**temp)

max_epochs = epochs
epochs = int(max_epochs/loc_ratio)
local_rounds = loc_ratio

results_path = os.path.dirname(os.path.abspath(__file__)) + f"/results/fl/{model_name}/{num_clients}_clients"
os.makedirs(results_path, exist_ok=True)
results_path = results_path+f"/metrics_{n_pred}model.csv"

checkpoint_path = os.path.dirname(os.path.abspath(__file__)) + f"/checkpoints/fl/{model_name}/{num_clients}_clients"
os.makedirs(checkpoint_path, exist_ok=True)
checkpoint_path = checkpoint_path + f"/{model_name}/{num_clients}_clients"

# load datasets and adjacency matrix
if city == 'chengdu':
    city_data, synthetic_data, data_mean, data_std, adj_mx = load_chengdu_data_new(f"data/chengdu/{num_clients}_client/inflow_main.npy",
         f"data/chengdu/{num_clients}_client/480_main.pkl")

if city == 'xian':
    city_data, synthetic_data, data_mean, data_std, adj_mx = load_chengdu_data_new(f"data/{city}/{num_clients}_client/inflow_main.npy",
     f"data/xian/{num_clients}_client/480_main.pkl")


train_data, eval_data, test_data = [],[],[]


for i in range(num_clients):
    train, evaluate, test = data_loader(city_data[i])
    train_data.append(train)
    eval_data.append(evaluate)
    test_data.append(test)

#transform data in above form into correct format for window and n_pred hyperparameters
training_data = [data_transform(train_data[i], window, n_pred, device) for i in range(num_clients)]
evaluation_data = [data_transform(eval_data[i], window, n_pred, device) for i in range(num_clients)]
testing_data = [data_transform(test_data[i], window, n_pred, device) for i in range(num_clients)]

# load generated data from DiffTraj Model
# process generated data into regional flows within interval, then append to training data for each client
if FL_method == "FedTPS":
    training_data = []
    for i in range(num_clients):
        x_synth, y_synth = data_transform(synthetic_data[0], window, n_pred, device)
        x_train, y_train = data_transform(train_data[i], window, n_pred, device)
        x_train = torch.cat((x_train, x_synth), dim=0)
        y_train = torch.cat((y_train, y_synth), dim=0)
        training_data.append([x_train, y_train])

#turn into pytorch TensorDataset type
train_tensor = [torch.utils.data.TensorDataset(partition[0], partition[1]) for partition in training_data]
eval_tensor = [torch.utils.data.TensorDataset(partition[0], partition[1]) for partition in evaluation_data]
test_tensor = [torch.utils.data.TensorDataset(partition[0], partition[1]) for partition in testing_data]

#turn datasets into pytorch dataloader type
trainloaders = [torch.utils.data.DataLoader(dataset, batch_size, shuffle = True, drop_last=True) for dataset in train_tensor]
evalloaders = [torch.utils.data.DataLoader(dataset, batch_size, shuffle = True, drop_last=True) for dataset in eval_tensor]
testloaders = [torch.utils.data.DataLoader(dataset, batch_size, shuffle = True, drop_last=True) for dataset in test_tensor]

test_iter = testloaders


#save metrics variable to which you will append results to
metric_results = {
    'n_pred':[n_pred],
    'train_losses':[[]],
    'test_mae':[],
    'test_rmse':[],
    'test_mape':[],
    'eval_metrics':[[]]
}

if model_name == "STGCN":
    # from adjacency matrix, produce graph for STGCN
    sparse_adjacency_matrix = sp.coo_matrix(adj_mx)
    G = dgl.from_scipy(sparse_adjacency_matrix)
    G = G.to(device)

    # function which allows flower to instantiate new clients
    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Load model
        #model instantiation
        model = STGCN(
            config.STGCN.blocks, window, config.STGCN.kt, config.STGCN.n, G, device, config.STGCN.drop_prob, control_str="TSTNDTSTND"
        ).to(device)
        lr=config.STGCN.lr

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        evalloader = evalloaders[int(cid)]
        testloader = testloaders
        client_mean = data_mean[int(cid)]
        client_std = data_std[int(cid)]

        # Create a  single Flower client representing a single organization

        if FL_method == "FedAvg" or FL_method == "FedProx" or FL_method == "FedTPS" or FL_method == "FedOpt":
            return FlowerClient(model, trainloader, evalloader, testloader, client_mean, client_std, local_rounds, lr=lr).to_client()
        else:
            print("FL method not defined")

    # model instantiation
    model = STGCN(
        config.STGCN.blocks, window, config.STGCN.kt, config.STGCN.n, G, device, config.STGCN.drop_prob, control_str="TSTNDTSTND"
    ).to(device)

elif model_name == "DCRNN":
    # process adjacency matrix, DCRNN also uses the doubletransition. Assuming asymmetric adjacency matrix.
    # takes one adjacency matrix
    adj_mx = load_adj(adj_mx, 'doubletransition')

    # function which allows flower to instantiate new clients
    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # model instantiation
        model = DCRNN(device, num_nodes=config.DCRNN.num_nodes, input_dim=config.DCRNN.input_dim, out_horizon=config.DCRNN.output_dim, P=adj_mx).to(device)
        lr=config.DCRNN.lr


        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        evalloader = evalloaders[int(cid)]
        client_mean = data_mean[int(cid)]
        client_std = data_std[int(cid)]
        testloader = testloaders

        if FL_method == "FedAvg" or FL_method == "FedProx" or FL_method == "FedTPS" or FL_method == "FedOpt":
            return FlowerClient(model, trainloader, evalloader, testloader, client_mean, client_std, local_rounds, lr=lr).to_client()
        else:
            print("FL method not defined")

    # load model for server side parameter evaluation
    # model instantiation
    model = DCRNN(device, num_nodes=config.DCRNN.num_nodes, input_dim=config.DCRNN.input_dim, out_horizon=config.DCRNN.output_dim, P=adj_mx).to(device)

elif model_name == "GWNET":
    # process adjacency matrix
    adj_mx = load_adj(adj_mx, 'doubletransition')
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    # random adjacency matrix initialisation
    adjinit = None


    # function which allows flower to instantiate new clients
    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        model = GWNET(device, n, config.GWNET.drop_prob, supports=supports, gcn_bool=config.GWNET.gcn_bool, addaptadj=config.GWNET.addaptadj, aptinit=adjinit, in_dim=config.GWNET.in_dim,
        out_dim=config.GWNET.out_dim, residual_channels=config.GWNET.n_hid, dilation_channels=config.GWNET.n_hid, skip_channels=config.GWNET.n_hid * 8, end_channels=config.GWNET.n_hid * 16).to(device)
        lr=1e-3


        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        evalloader = evalloaders[int(cid)]
        client_mean = data_mean[int(cid)]
        client_std = data_std[int(cid)]
        testloader = testloaders

        if FL_method == "FedAvg" or FL_method == "FedProx" or FL_method == "FedTPS" or FL_method == "FedOpt":
            return FlowerClient(model, trainloader, evalloader, testloader, client_mean, client_std, local_rounds, lr=lr).to_client()
        else:
            print("FL method not defined")

    # load model for server side parameter evaluation
    # model instantiation
    model = GWNET(device, n, config.GWNET.drop_prob, supports=supports, gcn_bool=config.GWNET.gcn_bool, addaptadj=config.GWNET.addaptadj,
                  aptinit=adjinit, in_dim=config.GWNET.in_dim, out_dim=config.GWNET.out_dim, residual_channels=config.GWNET.n_hid,
                  dilation_channels=config.GWNET.n_hid, skip_channels=config.GWNET.n_hid * 8, end_channels=config.GWNET.n_hid * 16).to(device)

elif model_name == "GRU":
    # function which allows flower to instantiate new clients
    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # model instantiation
        lr=1e-3

        model = GRU(config.GRU.input_size, config.GRU.hidden_size, config.GRU.num_layers, config.GRU.output_size, batch_size, num_nodes, window).to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        evalloader = evalloaders[int(cid)]
        client_mean = data_mean[int(cid)]
        client_std = data_std[int(cid)]
        testloader = testloaders

        if FL_method == "FedAvg" or FL_method == "FedProx" or FL_method == "FedTPS" or FL_method == "FedOpt":
            return FlowerClient(model, trainloader, evalloader, testloader, client_mean, client_std, local_rounds, lr=lr).to_client()
        else:
            print("FL method not defined")

    # load model for server side parameter evaluation
    # model instantiation
    num_nodes = 100

    model = GRU(config.GRU.input_size, config.GRU.hidden_size, config.GRU.num_layers, config.GRU.output_size, batch_size, num_nodes, window).to(device)

elif model_name == "TAU":
        # function which allows flower to instantiate new clients
    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        lr = 1e-3

        model = SimVP(config.TAU.in_shape, n_pred, config.TAU.hid_S, config.TAU.hid_T, config.TAU.N_S, config.TAU.N_T,
        model_type=config.TAU.model_type, mlp_ratio=8., drop=0.0, drop_path=config.TAU.drop_path, spatio_kernel_enc=config.TAU.spatio_kernel_enc,
        spatio_kernel_dec=config.TAU.spatio_kernel_dec, act_inplace=True).to(device)


        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        evalloader = evalloaders[int(cid)]
        client_mean = data_mean[int(cid)]
        client_std = data_std[int(cid)]
        testloader = testloaders

        if FL_method == "FedAvg" or FL_method == "FedProx" or FL_method == "FedTPS" or FL_method == "FedOpt":
            return FlowerClient(model, trainloader, evalloader, testloader, client_mean, client_std, local_rounds, lr=lr).to_client()
        else:
            print("FL method not defined")

    method = 'TAU'
    spatio_kernel_enc = 3
    spatio_kernel_dec = 3
    model_type = "tau"
    hid_S = 32
    hid_T = 256
    N_T = 8
    N_S = 2
    lr = 5e-4
    drop_path = 0.1
    sched = 'cosine'
    warmup_epoch = 5
    in_shape = (12,1,10,10)

    model = SimVP(config.TAU.in_shape, n_pred, config.TAU.hid_S, config.TAU.hid_T, config.TAU.N_S, config.TAU.N_T, model_type=config.TAU.model_type,
                mlp_ratio=8., drop=0.0, drop_path=config.TAU.drop_path, spatio_kernel_enc=config.TAU.spatio_kernel_enc,
                spatio_kernel_dec=config.TAU.spatio_kernel_dec, act_inplace=True).to(device)

elif model_name == "GATAU":
    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        lr = 1e-4

        sparse_adjacency_matrix = sp.coo_matrix(adj_mx)
        G = dgl.from_scipy(sparse_adjacency_matrix)
        G = G.to(device)


        model = GATAU(config.GATAU.in_shape, n_pred, G, config.GATAU.hid_S, config.GATAU.hid_T, config.GATAU.N_S, config.GATAU.N_T, model_type='tau',
                mlp_ratio=8., drop=0.0, drop_path=config.GATAU.drop_path, spatio_kernel_enc=config.GATAU.spatio_kernel_enc,
                spatio_kernel_dec=config.GATAU.spatio_kernel_dec, act_inplace=True).to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        evalloader = evalloaders[int(cid)]
        client_mean = data_mean[int(cid)]
        client_std = data_std[int(cid)]
        testloader = testloaders

        if FL_method == "FedAvg" or FL_method == "FedProx" or FL_method == "FedTPS" or FL_method == "FedOpt":
            return FlowerClient(model, trainloader, evalloader, testloader, client_mean, client_std, local_rounds, lr=lr).to_client()
        else:
            print("FL method not defined")

    sparse_adjacency_matrix = sp.coo_matrix(adj_mx)
    G = dgl.from_scipy(sparse_adjacency_matrix)
    G = G.to(device)

    method = 'TAU'
    # model
    spatio_kernel_enc = 3
    spatio_kernel_dec = 3
    model_type = "tau"
    hid_S = 64
    hid_T = 256
    N_T = 8
    N_S = 2
    # training
    lr = 1e-3
    drop_path = 0.1
    sched = 'cosine'
    warmup_epoch = 5
    in_shape = (12,1,10,10)

    model = GATAU(config.GATAU.in_shape, n_pred, G, config.GATAU.hid_S, config.GATAU.hid_T, config.GATAU.N_S,
                  config.GATAU.N_T, model_type='tau',
                  mlp_ratio=8., drop=0.0, drop_path=config.GATAU.drop_path,
                  spatio_kernel_enc=config.GATAU.spatio_kernel_enc,
                  spatio_kernel_dec=config.GATAU.spatio_kernel_dec, act_inplace=True).to(device)


# code below specifies flower config for FL algorithm used, and whether pre-trained model will be loaded
if initial_params == False:
    if FL_method == "FedAvg" or FL_method == "FedBN" or FL_method == "FedTPS":
        # FedAvg strategy
        strategy = FedAvgSave(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            checkpoint_path = checkpoint_path,
            n_pred = n_pred,
            epochs = epochs,
            model = model,
            test_iter = test_iter,
            data_mean = data_mean,
            data_std = data_std,
            metric_results = metric_results
        )

    elif FL_method == "FedProx":
        ## FedProx strategy
        strategy = FedProxSave(
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
            min_fit_clients=num_clients,  # Never sample less than all clients for training
            min_evaluate_clients=num_clients/2,  # Never sample less than half clients for evaluation
            min_available_clients=num_clients,  # Wait until all clients are available
            checkpoint_path = checkpoint_path,     #checkpoint path to save model parameters
            n_pred = n_pred,              #used to save model to correct file
            epochs = epochs,                 #used to save model at the end of training
            model = model,                   #pass model for final step centralised evaluation
            test_iter = test_iter,
            data_mean = data_mean,
            data_std = data_std,
            metric_results = metric_results,
            proximal_mu = 0.001
        )

    elif FL_method == "FedOpt":
        params = get_parameters(model)
        params = fl.common.ndarrays_to_parameters(params)
        strategy = FedOptSave(
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
            min_fit_clients=num_clients,  # Never sample less than all clients for training
            min_evaluate_clients=num_clients/2,  # Never sample less than half clients for evaluation
            min_available_clients=num_clients,  # Wait until all clients are available
            checkpoint_path = checkpoint_path,     #checkpoint path to save model parameters
            n_pred = n_pred,              #used to save model to correct file
            epochs = epochs,                 #used to save model at the end of training
            model = model,                   #pass model for final step centralised evaluation
            test_iter = test_iter,
            data_mean = data_mean,
            data_std = data_std,
            metric_results = metric_results,
            initial_parameters = params,
            eta = 1e-2, 
            eta_l = 0, 
            beta_1 = 0.9,
            beta_2 = 0.99,
        )

elif initial_params == True:
    print("Using initial parameters from pre-training!")

    state_dict = torch.load(f"checkpoints/synthetic/{city}/{model_name}/{num_clients}_clients/best_model_6pred.pt")
    model.load_state_dict(state_dict)
    state_dict_ndarrays = [v.cpu().numpy() for v in model.state_dict().values()]
    parameters = fl.common.ndarrays_to_parameters(state_dict_ndarrays)

    strategy = FedAvgSave(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
        min_fit_clients=num_clients,  # Never sample less than all clients for training
        min_evaluate_clients=num_clients/2,  # Never sample less than half clients for evaluation
        min_available_clients=num_clients,  # Wait until all clients are available
        checkpoint_path = checkpoint_path,     #checkpoint path to save model parameters
        n_pred = n_pred,              #used to save model to correct file
        epochs = epochs,                 #used to save model at the end of training
        model = model,                   #pass model for final step centralised evaluation
        test_iter = test_iter,
        data_mean = data_mean,
        data_std = data_std,
        metric_results = metric_results,
        initial_parameters = parameters
    )
else:
    print("Strategy not defined")

# Specify client resources if you need GPU for flower
client_resources = None
if device.type == "cuda":
    client_resources = {"num_gpus": 1}

# Start flower simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=num_clients,
    config=fl.server.ServerConfig(num_rounds=epochs),
    strategy=strategy,
    client_resources=client_resources,
)

# write metric results to csv file at results_path
if city == 'chengdu':

    df = pd.DataFrame.from_dict(metric_results)
    df.to_csv(results_path)
    file_name = 'chengdu_results.csv'
    row_contents = [city, num_clients, gen_samples, model_name, FL_method, local_rounds, epochs, metric_results['test_mae'][0], metric_results['test_mape'][0]*100]

    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(row_contents)

elif city == 'xian':

    df = pd.DataFrame.from_dict(metric_results)
    df.to_csv(results_path)
    file_name = 'xian_results.csv'
    row_contents = [city, num_clients, model_name, FL_method, local_rounds, epochs, metric_results['test_mae'][0], metric_results['test_mape'][0]*100]

    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(row_contents)

else:
    print("City not defined")