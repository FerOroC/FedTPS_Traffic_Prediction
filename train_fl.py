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

import argparse 

from models.gru import GRU
from models.dcrnn import DCRNN
from models.gwnet import GWNET
from models.stgcn import STGCN
from models.tau import SimVP
from models.mod_gat import Mod_gat
from models.mgat_tau import ASTGCN
from csv import writer

from utils.utils import load_chengdu_data_new, data_loader, data_transform, evaluate_mape, evaluate_model, load_adj
from utils.fl_utils import set_parameters, get_parameters, FlowerClient, FedAvgSave, FedProxSave, FedOptSave, train, test, FedBNFlowerClient



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
parser.add_argument("--loc_ratio", type=int)
parser.add_argument("--initial_params", type=int)
args = vars(parser.parse_args())

num_clients = args['num_clients']
model_name = args['model_name']
epochs = args['epochs']
city = args['city']
FL_method = args['FL_method']
loc_ratio = args['loc_ratio']
initial_params = bool(args['initial_params'])

max_epochs = epochs
epochs = int(max_epochs/loc_ratio)
local_rounds = loc_ratio

results_path = os.path.dirname(os.path.abspath(__file__)) +"/results/fl"
result_directories = [file for file in os.listdir(results_path)]

if model_name in result_directories:
    client_cases = [file for file in os.listdir(results_path + f"/{model_name}")]
    if not(f"{num_clients}_clients" in client_cases):
        model_client_path = results_path + f"/{model_name}/{num_clients}_clients"
        os.mkdir(model_client_path)
        results_path = model_client_path+f"/metrics_{n_pred}model.csv"
    model_client_path = results_path + f"/{model_name}/{num_clients}_clients"
    results_path = model_client_path+f"/metrics_{n_pred}model.csv"
else:
    os.mkdir(results_path + f"/{model_name}")
    model_client_path = results_path + f"/{model_name}/{num_clients}_clients"
    os.mkdir(model_client_path)
    results_path = model_client_path+f"/metrics_{n_pred}model.csv"

checkpoint_path = os.path.dirname(os.path.abspath(__file__)) +"/checkpoints/fl"
checkpoint_directories = [file for file in os.listdir(checkpoint_path)]
if model_name in checkpoint_directories:
    client_cases = [file for file in os.listdir(checkpoint_path + f"/{model_name}")]
    if not(f"{num_clients}_clients" in client_cases):
        checkpoint_path = checkpoint_path + f"/{model_name}/{num_clients}_clients"
        os.mkdir(checkpoint_path)
    checkpoint_path = checkpoint_path + f"/{model_name}/{num_clients}_clients"
else:
    os.mkdir(checkpoint_path + f"/{model_name}")
    checkpoint_path = checkpoint_path + f"/{model_name}/{num_clients}_clients"
    os.mkdir(checkpoint_path)
print("Checkpoint path: ", checkpoint_path)

# load datasets and adjacency matrix
#40-60
city_data, synthetic_data, data_mean, data_std, adj_mx = load_chengdu_data_new(f"data/{num_clients}_client_chengdu_inflow.npy",
     f"data/{num_clients}_client_chengdu_synth.npy")
print("mean data: ", np.mean(data_mean))

if city == 'xian':
    city_data, synthetic_data, data_mean, data_std, adj_mx = load_chengdu_data_new(f"data/xian/{num_clients}_client_xian_inflow.npy",
     f"data/xian/{num_clients}_client_xian_synth.npy")

    print("Mean data: ", data_mean)

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

# generic synthetic data, for even distribution like 50
# load generated data from DiffTraj Model
# print("Synthetic data shape: ", synthetic_data.shape)
# synthetic_data_transformed = data_transform(synthetic_data, window, n_pred, device)
# train_synthetic_data = torch.utils.data.TensorDataset(synthetic_data_transformed[0], synthetic_data_transformed[1])
# train_synthetic_loader = torch.utils.data.DataLoader(train_synthetic_data, 1, shuffle = True, drop_last = True)
# training_data = []
# for i in range(num_clients):
#     x_train, y_train = data_transform(train_data[i], window, n_pred, device)
#     x_train = torch.cat((x_train, synthetic_data_transformed[0]), dim=0)
#     print("X train shape: ", x_train.shape)
#     y_train = torch.cat((y_train, synthetic_data_transformed[1]), dim=0)
#     training_data.append([x_train, y_train])

if FL_method == "FedSTS":
    #client specific synthetic data, for uneven distributions like 40-60
    training_data = []
    for i in range(num_clients):
        x_synth, y_synth = data_transform(synthetic_data[0], window, n_pred, device)
        x_train, y_train = data_transform(train_data[i], window, n_pred, device)
        x_train = torch.cat((x_train, x_synth), dim=0)
        print("X train shape: ", x_train.shape)
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

#just using one part of testloaders for centralised model testing
test_iter = testloaders

print("Shape of training data: ", training_data[0][0].shape)


#save metrics variable to which you will append results to
metric_results = {
    'n_pred':[n_pred],
    'train_losses':[[]],
    'test_mae':[],
    'test_rmse':[],
    'test_mape':[],
    'eval_metrics':[[]]
}

if model_name == "MGAT_TAU":
        # from adjacency matrix, produce graph for STGCN
    sparse_adjacency_matrix = sp.coo_matrix(adj_mx)
    G = dgl.from_scipy(sparse_adjacency_matrix)
    G = G.to(device)

    # function which allows flower to instantiate new clients
    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Load model
        #model instantiation
        nb_block = 2
        in_channels = 1
        K = 3
        nb_chev_filters = 64
        nb_time_filters = 64
        time_strides = 0
        edge_index = G.edges()
        num_for_predict=1
        len_input = 12
        num_of_vertices = len(G.edges()[0])

        model = ASTGCN(nb_block, in_channels, K, nb_chev_filters, nb_time_filters, time_strides, edge_index, num_for_predict, len_input, num_of_vertices).to(device)
        lr=1e-3

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        evalloader = evalloaders[int(cid)]
        testloader = testloaders
        client_mean = data_mean[int(cid)]
        client_std = data_std[int(cid)]

        # Create a  single Flower client representing a single organization

        # Different client call if FedBN called
        if FL_method == "FedBN":
            return FedBNFlowerClient(model=model, trainloader=trainloader, valloader=evalloader, testloader=testloader, 
            data_mean=client_mean, data_std=client_std, local_rounds=local_rounds, save_path=f"FedBN/{model_name}/{num_clients}_clients", client_id=cid, lr=lr)
        elif FL_method == "FedAvg" or FL_method == "FedProx" or FL_method == "FedSTS" or FL_method == "FedOpt":
            return FlowerClient(model, trainloader, evalloader, testloader, client_mean, client_std, local_rounds, lr=lr).to_client()
        else:
            print("FL method not defined")

    # Load model
    #model instantiation
    nb_block = 2
    in_channels = 1
    K = 3
    nb_chev_filters = 64
    nb_time_filters = 64
    time_strides = 0
    edge_index = G.edges()
    num_for_predict=1
    len_input = 12
    num_of_vertices = len(G.edges()[0])

    model = ASTGCN(nb_block, in_channels, K, nb_chev_filters, nb_time_filters, time_strides, edge_index, num_for_predict, len_input, num_of_vertices).to(device)

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
        blocks = [1, 64, 64, 64, 64, 64, 32, 32, 128, 128]
        drop_prob = 0.3
        n=100
        kt=3   #kernel size conv
        model = STGCN(
            blocks, window, kt, n, G, device, drop_prob, control_str="TSTNDTSTND"
        ).to(device)
        lr=1e-3

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        evalloader = evalloaders[int(cid)]
        testloader = testloaders
        client_mean = data_mean[int(cid)]
        client_std = data_std[int(cid)]

        # Create a  single Flower client representing a single organization

        # Different client call if FedBN called
        if FL_method == "FedBN":
            return FedBNFlowerClient(model=model, trainloader=trainloader, valloader=evalloader, testloader=testloader, 
            data_mean=client_mean, data_std=client_std, local_rounds=local_rounds, save_path=f"FedBN/{model_name}/{num_clients}_clients", client_id=cid, lr=lr)
        elif FL_method == "FedAvg" or FL_method == "FedProx" or FL_method == "FedSTS" or FL_method == "FedOpt":
            return FlowerClient(model, trainloader, evalloader, testloader, client_mean, client_std, local_rounds, lr=lr).to_client()
        else:
            print("FL method not defined")

    # model instantiation
    blocks = [1, 64, 64, 64, 64, 64, 32, 32, 128, 128]
    drop_prob = 0.3
    n=100
    kt=3   #kernel size conv
    model = STGCN(
        blocks, window, kt, n, G, device, drop_prob, control_str="TSTNDTSTND"
    ).to(device)

elif model_name == "DCRNN":
    # process adjacency matrix, DCRNN also uses the doubletransition. Assuming asymmetric adjacency matrix.
    # takes one adjacency matrix
    adj_mx = load_adj(adj_mx, 'doubletransition')

    # function which allows flower to instantiate new clients
    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # model instantiation
        input_dim = 1
        num_nodes = 100
        output_dim = 1
        model = DCRNN(device, num_nodes=num_nodes, input_dim=input_dim, out_horizon=output_dim, P=adj_mx).to(device)
        lr=1e-3


        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        evalloader = evalloaders[int(cid)]
        client_mean = data_mean[int(cid)]
        client_std = data_std[int(cid)]
        testloader = testloaders

        # Different client call if FedBN called
        if FL_method == "FedBN":
            return FedBNFlowerClient(model=model, trainloader=trainloader, valloader=evalloader, testloader=testloader, 
            data_mean=client_mean, data_std=client_std, local_rounds=local_rounds, save_path=f"FedBN/{model_name}/{num_clients}_clients", client_id=cid, lr=lr)
        elif FL_method == "FedAvg" or FL_method == "FedProx" or FL_method == "FedSTS" or FL_method == "FedOpt":
            return FlowerClient(model, trainloader, evalloader, testloader, client_mean, client_std, local_rounds, lr=lr).to_client()
        else:
            print("FL method not defined")

    # load model for server side parameter evaluation
    # model instantiation
    input_dim = 1
    num_nodes = 100
    output_dim = 1
    model = DCRNN(device, num_nodes=num_nodes, input_dim=input_dim, out_horizon=output_dim, P=adj_mx).to(device)

elif model_name == "GWNET":
    # process adjacency matrix
    adj_mx = load_adj(adj_mx, 'doubletransition')
    np.set_printoptions(threshold=np.inf)
    print(adj_mx[0])
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    # random adjacency matrix initialisation
    adjinit = None

    drop_prob = 0.3

    # function which allows flower to instantiate new clients
    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # model instantiation
        gcn_bool = True              # add a graph conv layer
        addaptadj = True             # add an adaptive adjacency matrix
        in_dim = 1                   # dimension of channel size of input 
        out_dim = 1                  # dimension fo channel size of output, this changes when using multi-step prediction
        n_hid = 32                   # dimension of hidden residual channel
        model = GWNET(device, n, drop_prob, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=adjinit, in_dim=in_dim, 
        out_dim=out_dim, residual_channels=n_hid, dilation_channels=n_hid, skip_channels=n_hid * 8, end_channels=n_hid * 16).to(device)
        lr=1e-3


        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        evalloader = evalloaders[int(cid)]
        client_mean = data_mean[int(cid)]
        client_std = data_std[int(cid)]
        testloader = testloaders

        # Different client call if FedBN called
        if FL_method == "FedBN":
            return FedBNFlowerClient(model=model, trainloader=trainloader, valloader=evalloader, testloader=testloader, 
            data_mean=client_mean, data_std=client_std, local_rounds=local_rounds, save_path=f"FedBN/{model_name}/{num_clients}_clients", client_id=cid, lr=lr)
        elif FL_method == "FedAvg" or FL_method == "FedProx" or FL_method == "FedSTS" or FL_method == "FedOpt":
            return FlowerClient(model, trainloader, evalloader, testloader, client_mean, client_std, local_rounds, lr=lr).to_client()
        else:
            print("FL method not defined")

    # load model for server side parameter evaluation
    # model instantiation
    gcn_bool = True              # add a graph conv layer
    addaptadj = True             # add an adaptive adjacency matrix
    in_dim = 1                   # dimension of channel size of input 
    out_dim = 1                  # dimension fo channel size of output, this changes when using multi-step prediction
    n_hid = 32                   # dimension of hidden residual channel
    model = GWNET(device, n, drop_prob, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=adjinit, in_dim=in_dim, 
    out_dim=out_dim, residual_channels=n_hid, dilation_channels=n_hid, skip_channels=n_hid * 8, end_channels=n_hid * 16).to(device)

elif model_name == "GRU":
    # function which allows flower to instantiate new clients
    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # model instantiation
        hidden_size = 32
        num_layers = 3
        output_size = 1
        input_size = 1
        num_nodes = 100
        lr=1e-3


        model = GRU(input_size, hidden_size, num_layers, output_size, batch_size, num_nodes, window).to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        evalloader = evalloaders[int(cid)]
        client_mean = data_mean[int(cid)]
        client_std = data_std[int(cid)]
        testloader = testloaders

        # Different client call if FedBN called
        if FL_method == "FedBN":
            return FedBNFlowerClient(model=model, trainloader=trainloader, valloader=evalloader, testloader=testloader, 
            data_mean=client_mean, data_std=client_std, local_rounds=local_rounds, save_path=f"FedBN/{model_name}/{num_clients}_clients", client_id=cid, lr=lr)
        elif FL_method == "FedAvg" or FL_method == "FedProx" or FL_method == "FedSTS" or FL_method == "FedOpt":
            return FlowerClient(model, trainloader, evalloader, testloader, client_mean, client_std, local_rounds, lr=lr).to_client()
        else:
            print("FL method not defined")

    # load model for server side parameter evaluation
    # model instantiation
    hidden_size = 32
    num_layers = 3
    output_size = 1
    input_size = 1
    num_nodes = 100

    model = GRU(input_size, hidden_size, num_layers, output_size, batch_size, num_nodes, window).to(device)

elif model_name == "TAU":
        # function which allows flower to instantiate new clients
    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        method = 'TAU'
        # model
        spatio_kernel_enc = 3
        spatio_kernel_dec = 3
        model_type = "tau"
        hid_S = 32
        hid_T = 256
        N_T = 8
        N_S = 2
        # training
        lr = 1e-3
        drop_path = 0.1
        sched = 'cosine'
        warmup_epoch = 5
        in_shape = (12,1,10,10)

        model = SimVP(in_shape, n_pred, hid_S, hid_T, N_S, N_T, model_type=model_type, mlp_ratio=8., drop=0.0, drop_path=drop_path, spatio_kernel_enc=spatio_kernel_enc,
                spatio_kernel_dec=spatio_kernel_dec, act_inplace=True).to(device)


        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        evalloader = evalloaders[int(cid)]
        client_mean = data_mean[int(cid)]
        client_std = data_std[int(cid)]
        testloader = testloaders

        # Different client call if FedBN called
        if FL_method == "FedBN":
            return FedBNFlowerClient(model=model, trainloader=trainloader, valloader=evalloader, testloader=testloader, 
            data_mean=client_mean, data_std=client_std, local_rounds=local_rounds, save_path=f"FedBN/{model_name}/{num_clients}_clients", client_id=cid, lr=lr)
        elif FL_method == "FedAvg" or FL_method == "FedProx" or FL_method == "FedSTS" or FL_method == "FedOpt":
            return FlowerClient(model, trainloader, evalloader, testloader, client_mean, client_std, local_rounds, lr=lr).to_client()
        else:
            print("FL method not defined")

    method = 'TAU'
    # model
    spatio_kernel_enc = 3
    spatio_kernel_dec = 3
    model_type = "tau"
    hid_S = 32
    hid_T = 256
    N_T = 8
    N_S = 2
    # training
    lr = 5e-4
    drop_path = 0.1
    sched = 'cosine'
    warmup_epoch = 5
    in_shape = (12,1,10,10)

    model = SimVP(in_shape, n_pred, hid_S, hid_T, N_S, N_T, model_type=model_type, mlp_ratio=8., drop=0.0, drop_path=drop_path, spatio_kernel_enc=spatio_kernel_enc,
            spatio_kernel_dec=spatio_kernel_dec, act_inplace=True).to(device)

elif model_name == "MOD_GAT":
        # function which allows flower to instantiate new clients
    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        method = 'TAU'
        # model
        spatio_kernel_enc = 3
        spatio_kernel_dec = 3
        model_type = "tau"
        hid_S = 16
        hid_T = 256
        N_T = 8
        N_S = 2
        # training
        lr = 1e-4
        drop_path = 0.1
        sched = 'cosine'
        warmup_epoch = 5
        in_shape = (12,1,10,10)

        sparse_adjacency_matrix = sp.coo_matrix(adj_mx)
        G = dgl.from_scipy(sparse_adjacency_matrix)
        G = G.to(device)

        # SINGLE LAYER

        # lr 5e-4, hid_S 32, tau, MAPE 80
        # lr 1e-3, hid_S 32, tau, MAPE >50
        # lr 1e-3, hid_S 16, tau, MAPE <50

        #DOUBLE LAYER 


        model = Mod_gat(in_shape, n_pred, G, hid_S, hid_T, N_S, N_T, model_type=model_type, mlp_ratio=8., drop=0.0, drop_path=drop_path, spatio_kernel_enc=spatio_kernel_enc,
                 spatio_kernel_dec=spatio_kernel_dec, act_inplace=True).to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        evalloader = evalloaders[int(cid)]
        client_mean = data_mean[int(cid)]
        client_std = data_std[int(cid)]
        testloader = testloaders

        # Different client call if FedBN called
        if FL_method == "FedBN":
            return FedBNFlowerClient(model=model, trainloader=trainloader, valloader=evalloader, testloader=testloader, 
            data_mean=client_mean, data_std=client_std, local_rounds=local_rounds, save_path=f"FedBN/{model_name}/{num_clients}_clients", client_id=cid, lr=lr)
        elif FL_method == "FedAvg" or FL_method == "FedProx" or FL_method == "FedSTS" or FL_method == "FedOpt":
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
    hid_S = 16
    hid_T = 256
    N_T = 8
    N_S = 2
    # training
    lr = 1e-3
    drop_path = 0.1
    sched = 'cosine'
    warmup_epoch = 5
    in_shape = (12,1,10,10)

    model = Mod_gat(in_shape, n_pred, G, hid_S, hid_T, N_S, N_T, model_type=model_type, mlp_ratio=8., drop=0.0, drop_path=drop_path, spatio_kernel_enc=spatio_kernel_enc,
                 spatio_kernel_dec=spatio_kernel_dec, act_inplace=True).to(device)

if initial_params == False:
    if FL_method == "FedAvg" or FL_method == "FedBN" or FL_method == "FedSTS":
        # FedAvg strategy
        strategy = FedAvgSave(
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
            min_fit_clients=num_clients,  # Never sample less than all clients for training
            min_evaluate_clients=num_clients,  # Never sample less than half clients for evaluation
            min_available_clients=num_clients,  # Wait until all clients are available
            checkpoint_path = checkpoint_path,     #checkpoint path to save model parameters
            n_pred = n_pred,              #used to save model to correct file
            epochs = epochs,                 #used to save model at the end of training
            model = model,                   #pass model for final step centralised evaluation
            test_iter = test_iter,
            data_mean = data_mean,
            data_std = data_std,
            metric_results = metric_results
            #on_fit_config_fn
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
            #on_fit_config_fn
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
    state_dict = torch.load(f"checkpoints/synthetic/{model_name}/{num_clients}_clients/best_model_6pred.pt")
    model.load_state_dict(state_dict)
    state_dict_ndarrays = [v.cpu().numpy() for v in model.state_dict().values()]
    parameters = fl.common.ndarrays_to_parameters(state_dict_ndarrays)


    def model_summary(model):
        print("model_summary")
        print()
        print("Layer_name"+"\t"*7+"Number of Parameters")
        print("="*100)
        model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
        layer_name = [child for child in model.children()]
        j = 0
        total_params = 0
        print("\t"*10)
        for i in layer_name:
            print()
            param = 0
            try:
                bias = (i.bias is not None)
            except:
                bias = False  
            if not bias:
                param =model_parameters[j].numel()+model_parameters[j+1].numel()
                j = j+2
            else:
                param =model_parameters[j].numel()
                j = j+1
            print(str(i)+"\t"*3+str(param))
            total_params+=param
        print("="*100)
        print(f"Total Params:{total_params}")       

    model_summary(model)


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
        #on_fit_config_fn
    )
else:
    print("Strategy not defined")

# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
if device.type == "cuda":
    client_resources = {"num_gpus": 1}

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=num_clients,
    config=fl.server.ServerConfig(num_rounds=epochs),
    strategy=strategy,
    client_resources=client_resources,
)

if city == 'chengdu':
    print("metric_results: ", metric_results)
    df = pd.DataFrame.from_dict(metric_results)

    print(df)

    df.to_csv(results_path)

    file_name = 'results.csv'
    row_contents = [city, num_clients, model_name, FL_method, local_rounds, epochs, metric_results['test_mae'][0], 0,metric_results['test_mape'][0]]

    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(row_contents)
else:
    print("metric_results: ", metric_results)
    df = pd.DataFrame.from_dict(metric_results)

    print(df)

    df.to_csv(results_path)

    file_name = 'results_xian.csv'
    row_contents = [city, num_clients, model_name, FL_method, local_rounds, epochs, metric_results['test_mae'][0], metric_results['test_rmse'][0], metric_results['test_mape'][0]]

    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(row_contents)