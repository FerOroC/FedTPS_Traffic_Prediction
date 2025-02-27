import torch
import numpy as np
import scipy.sparse as sp
import dgl
import torch.nn as nn
import pandas as pd

from utils.utils import load_chengdu_data_new, data_loader, data_transform, evaluate_mape, evaluate_metric, \
    evaluate_model, train_model, load_adj, numpy_to_graph

from models.gru import GRU
from models.dcrnn import DCRNN
from models.gwnet import GWNET
from models.stgcn import STGCN
from models.tau import SimVP
from models.gatau import GATAU

from csv import writer

import argparse
import os

window = 12
n_pred = 6

lr = 0.001
batch_size = 10
drop_prob = 0.3
n = 100

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

if __name__ == '__main__':
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
    parser.add_argument("--synthetic", required=True, help="Using only synthetic data or not", type=str)
    args = vars(parser.parse_args())

    num_clients = args['num_clients']
    model_name = args['model_name']
    epochs = args['epochs']
    city = args['city']
    synthetic = args['synthetic']

    best_model_path = f"checkpoints/synthetic/{city}/{model_name}/{num_clients}_clients/best_model_{n_pred}pred.pt"



    if synthetic == 'yes':
        city_data, synthetic_data, data_mean, data_std, adj_mx = load_chengdu_data_new(
            f"data/{city}/{num_clients}_client/480_main.pkl",
            f"data/{city}/1_client/inflow_main.npy",)
    else:
        # load datasets and adjacency matrix
        city_data, synthetic_data, data_mean, data_std, adj_mx = load_chengdu_data_new(
            f"data/{city}/{num_clients}_client/inflow_main.npy",
            f"data/{city}/1_client/480_main.pkl")

    # split data into train, eval, test data
    train_data, eval_data, test_data = [], [], []

    for i in range(num_clients):
        train, evaluate, test = data_loader(city_data[i])
        train_data.append(train)
        eval_data.append(evaluate)
        test_data.append(test)

    training_data = [data_transform(train_data[i], window, n_pred, device) for i in range(num_clients)]
    evaluation_data = [data_transform(eval_data[i], window, n_pred, device) for i in range(num_clients)]
    testing_data = [data_transform(test_data[i], window, n_pred, device) for i in range(num_clients)]


    # turn into pytorch TensorDataset type
    train_tensor = [torch.utils.data.TensorDataset(partition[0], partition[1]) for partition in training_data]
    eval_tensor = [torch.utils.data.TensorDataset(partition[0], partition[1]) for partition in evaluation_data]
    test_tensor = [torch.utils.data.TensorDataset(partition[0], partition[1]) for partition in testing_data]

    # turn datasets into pytorch dataloader type
    trainloaders = [torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True) for dataset in
                    train_tensor]
    evalloaders = [torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True) for dataset in
                   eval_tensor]
    testloaders = [torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True) for dataset in
                   test_tensor]

    # save metrics variable to which you will append results to
    metric_results = {
        'n_pred': [n_pred],
        'train_losses': [[]],
        'test_mae': [],
        'test_rmse': [],
        'test_mape': []
    }

    # loading appropriate model, and training
    if model_name == "STGCN":
        # from adjacency matrix, produce graph for STGCN
        sparse_adjacency_matrix = sp.coo_matrix(adj_mx)
        G = dgl.from_scipy(sparse_adjacency_matrix)
        G = G.to(device)

        # train centralised model
        min_val_loss = np.inf
        loss = nn.MSELoss()
        best_train_losses = []
        best_client_id = 0
        for client_id, train_iter in enumerate(trainloaders):

            # model instantiation
            blocks = [1, 64, 64, 64, 64, 64, 32, 32, 128, 128]
            drop_prob = 0.3
            n = 100
            kt = 3  # kernel size conv
            model = STGCN(
                blocks, window, kt, n, G, device, drop_prob, control_str="TSTNDTSTND"
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

            client_model, client_val_loss, client_train_losses = train_model(model, epochs, train_iter, loss, optimizer,
                                                                             scheduler, evalloaders[client_id],
                                                                             data_mean[client_id], data_std[client_id])
            if client_val_loss < min_val_loss:
                print("Saving best client's model to: ", best_model_path)
                torch.save(client_model.state_dict(), best_model_path)
                best_train_losses = client_train_losses
                best_client_id = client_id

    elif model_name == "DCRNN":
        # process adjacency matrix, DCRNN also uses the doubletransition. Assuming asymmetric adjacency matrix.
        adj_mx = load_adj(adj_mx, 'doubletransition')

        # train centralised model
        min_val_loss = np.inf
        loss = nn.MSELoss()
        best_train_losses = []
        best_client_id = 0
        for client_id, train_iter in enumerate(trainloaders):

            # model instantiation
            input_dim = 1
            num_nodes = 100
            output_dim = 1
            model = DCRNN(device, num_nodes=num_nodes, input_dim=input_dim, out_horizon=output_dim, P=adj_mx).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

            client_model, client_val_loss, client_train_losses = train_model(model, epochs, train_iter, loss, optimizer,
                                                                             scheduler, evalloaders[client_id],
                                                                             data_mean[client_id], data_std[client_id])
            if client_val_loss < min_val_loss:
                print("Saving best client's model to: ", best_model_path)
                torch.save(client_model.state_dict(), best_model_path)
                best_train_losses = client_train_losses
                best_client_id = client_id


    elif model_name == "GWNET":
        # process adjacency matrix
        adj_mx = load_adj(adj_mx, 'doubletransition')
        supports = [torch.tensor(i).to(device) for i in adj_mx]

        # random adjacency matrix initialisation
        adjinit = None

        drop_prob = 0.3

        # train centralised model
        min_val_loss = np.inf
        loss = nn.MSELoss()
        best_train_losses = []
        best_client_id = 0
        for client_id, train_iter in enumerate(trainloaders):

            # model instantiation
            gcn_bool = True  # add a graph conv layer
            addaptadj = True  # add an adaptive adjacency matrix
            in_dim = 1  # dimension of channel size of input
            out_dim = 1  # dimension fo channel size of output, this changes when using multi-step prediction
            n_hid = 32  # dimension of hidden residual channel
            model = GWNET(device, n, drop_prob, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj,
                          aptinit=adjinit, in_dim=in_dim,
                          out_dim=out_dim, residual_channels=n_hid, dilation_channels=n_hid, skip_channels=n_hid * 8,
                          end_channels=n_hid * 16).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

            client_model, client_val_loss, client_train_losses = train_model(model, epochs, train_iter, loss, optimizer,
                                                                             scheduler, evalloaders[client_id],
                                                                             data_mean[client_id], data_std[client_id])
            if client_val_loss < min_val_loss:
                print("Saving best client's model to: ", best_model_path)
                torch.save(client_model.state_dict(), best_model_path)
                best_train_losses = client_train_losses
                best_client_id = client_id

    elif model_name == "GRU":
        # train centralised model
        min_val_loss = np.inf
        loss = nn.MSELoss()
        best_train_losses = []
        best_client_id = 0
        for client_id, train_iter in enumerate(trainloaders):

            # model instantiation
            hidden_size = 32
            num_layers = 2
            output_size = 1
            input_size = 1
            num_nodes = 100

            model = GRU(input_size, hidden_size, num_layers, output_size, batch_size, num_nodes, window).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

            client_model, client_val_loss, client_train_losses = train_model(model, epochs, train_iter, loss, optimizer,
                                                                             scheduler, evalloaders[client_id],
                                                                             data_mean[client_id], data_std[client_id])
            if client_val_loss < min_val_loss:
                print("Saving best client's model to: ", best_model_path)
                torch.save(client_model.state_dict(), best_model_path)
                best_train_losses = client_train_losses
                best_client_id = client_id

    elif model_name == "SimVP":
        min_val_loss = np.inf
        loss = nn.MSELoss()
        best_train_losses = []
        best_client_id = 0
        for client_id, train_iter in enumerate(trainloaders):

            method = 'SimVP'
            # model
            spatio_kernel_enc = 3
            spatio_kernel_dec = 3
            model_type = "gsta"
            hid_S = 32
            hid_T = 256
            N_T = 8
            N_S = 2
            # training
            lr = 1e-3
            drop_path = 0.1
            sched = 'cosine'
            warmup_epoch = 5
            in_shape = (12, 1, 10, 10)

            model = SimVP(in_shape, n_pred, hid_S, hid_T, N_S, N_T, model_type=model_type, mlp_ratio=8., drop=0.0,
                          drop_path=drop_path, spatio_kernel_enc=spatio_kernel_enc,
                          spatio_kernel_dec=spatio_kernel_dec, act_inplace=True).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

            client_model, client_val_loss, client_train_losses = train_model(model, epochs, train_iter, loss, optimizer,
                                                                             scheduler, evalloaders[client_id],
                                                                             data_mean[client_id], data_std[client_id])
            if client_val_loss < min_val_loss:
                print("Saving best client's model to: ", best_model_path)
                torch.save(client_model.state_dict(), best_model_path)
                best_train_losses = client_train_losses
                best_client_id = client_id

    elif model_name == "TAU":
        min_val_loss = np.inf
        loss = nn.MSELoss()
        best_train_losses = []
        best_client_id = 0
        for client_id, train_iter in enumerate(trainloaders):

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
            in_shape = (12, 1, 10, 10)

            model = SimVP(in_shape, n_pred, hid_S, hid_T, N_S, N_T, model_type=model_type, mlp_ratio=8., drop=0.0,
                          drop_path=drop_path, spatio_kernel_enc=spatio_kernel_enc,
                          spatio_kernel_dec=spatio_kernel_dec, act_inplace=True).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

            client_model, client_val_loss, client_train_losses = train_model(model, epochs, train_iter, loss, optimizer,
                                                                             scheduler, evalloaders[client_id],
                                                                             data_mean[client_id], data_std[client_id])
            if client_val_loss < min_val_loss:
                print("Saving best client's model to: ", best_model_path)
                torch.save(client_model.state_dict(), best_model_path)
                best_train_losses = client_train_losses
                best_client_id = client_id

    elif model_name == "GATAU":
        sparse_adjacency_matrix = sp.coo_matrix(adj_mx)
        G = dgl.from_scipy(sparse_adjacency_matrix)
        G = G.to(device)

        min_val_loss = np.inf
        loss = nn.MSELoss()
        best_train_losses = []
        best_client_id = 0
        for client_id, train_iter in enumerate(trainloaders):

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
            in_shape = (12, 1, 10, 10)
            local_rounds = 1

            sparse_adjacency_matrix = sp.coo_matrix(adj_mx)
            G = dgl.from_scipy(sparse_adjacency_matrix)
            G = G.to(device)

            model = GATAU(in_shape, n_pred, G, hid_S, hid_T, N_S, N_T, model_type=model_type, mlp_ratio=8., drop=0.0,
                          drop_path=drop_path, spatio_kernel_enc=spatio_kernel_enc,
                          spatio_kernel_dec=spatio_kernel_dec, act_inplace=True).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

            client_model, client_val_loss, client_train_losses = train_model(model, epochs, train_iter, loss, optimizer,
                                                                             scheduler, evalloaders[client_id],
                                                                             data_mean[client_id], data_std[client_id])
            if client_val_loss < min_val_loss:
                print("Saving best client's model to: ", best_model_path)
                torch.save(client_model.state_dict(), best_model_path)
                best_train_losses = client_train_losses
                best_client_id = client_id

    # Evaluation
    if model_name == "STGCN":
        # model instantiation
        blocks = [1, 64, 64, 64, 64, 64, 32, 32, 128, 128]
        drop_prob = 0.3
        n = 100
        kt = 3  # kernel size conv
        best_model = STGCN(
            blocks, window, kt, n, G, device, drop_prob, control_str="TSTNDTSTND"
        ).to(device)

        best_model.load_state_dict(torch.load(best_model_path))

        l = evaluate_model(best_model, loss, testloaders[best_client_id])
        # indexing first client data mean and std since test_iter is referenced as first client at data loading stage for simplification purposes
        MAE, RMSE, MAPE = evaluate_metric(best_model, testloaders[best_client_id], data_mean[0], data_std[0])
        print("test loss:", l, "\nMAE:", MAE, "RMSE:", RMSE, "MAPE:", MAPE * 100)

        metric_results['train_losses'][0].append(best_train_losses)
        metric_results['test_mae'].append(MAE)
        metric_results['test_rmse'].append(RMSE)
        metric_results['test_mape'].append(MAPE * 100)


    elif model_name == "DCRNN":

        # load model for server side parameter evaluation
        # model instantiation
        input_dim = 1
        num_nodes = 100
        output_dim = 1
        best_model = DCRNN(device, num_nodes=num_nodes, input_dim=input_dim, out_horizon=output_dim, P=adj_mx).to(
            device)

        best_model.load_state_dict(torch.load(best_model_path))

        l = evaluate_model(best_model, loss, testloaders[best_client_id])
        # indexing first client data mean and std since test_iter is referenced as first client at data loading stage for simplification purposes
        MAE, RMSE, MAPE = evaluate_metric(best_model, testloaders[best_client_id], data_mean[0], data_std[0])
        print("test loss:", l, "\nMAE:", MAE, "RMSE:", RMSE, "MAPE:", MAPE * 100)

        metric_results['train_losses'][0].append(best_train_losses)
        metric_results['test_mae'].append(MAE)
        metric_results['test_rmse'].append(RMSE)
        metric_results['test_mape'].append(MAPE * 100)

    elif model_name == "GWNET":

        # model instantiation
        gcn_bool = True  # add a graph conv layer
        addaptadj = True  # add an adaptive adjacency matrix
        in_dim = 1  # dimension of channel size of input
        out_dim = 1  # dimension fo channel size of output, this changes when using multi-step prediction
        n_hid = 32  # dimension of hidden residual channel
        best_model = GWNET(device, n, drop_prob, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj,
                           aptinit=adjinit, in_dim=in_dim,
                           out_dim=out_dim, residual_channels=n_hid, dilation_channels=n_hid, skip_channels=n_hid * 8,
                           end_channels=n_hid * 16).to(device)

        best_model.load_state_dict(torch.load(best_model_path))

        l = evaluate_model(best_model, loss, testloaders[best_client_id])
        # indexing first client data mean and std since test_iter is referenced as first client at data loading stage for simplification purposes
        MAE, RMSE, MAPE = evaluate_metric(best_model, testloaders[best_client_id], data_mean[0], data_std[0])
        print("test loss:", l, "\nMAE:", MAE, "RMSE:", RMSE, "MAPE:", MAPE * 100)

        metric_results['train_losses'][0].append(best_train_losses)
        metric_results['test_mae'].append(MAE)
        metric_results['test_rmse'].append(RMSE)
        metric_results['test_mape'].append(MAPE * 100)


    elif model_name == "GRU":
        # model instantiation
        hidden_size = 32
        num_layers = 2
        output_size = 1
        input_size = 1
        num_nodes = 100
        best_model = GRU(input_size, hidden_size, num_layers, output_size, batch_size, num_nodes, window).to(device)

        best_model.load_state_dict(torch.load(best_model_path))

        l = evaluate_model(best_model, loss, testloaders[best_client_id])
        # indexing first client data mean and std since test_iter is referenced as first client at data loading stage for simplification purposes
        MAE, RMSE, MAPE = evaluate_metric(best_model, testloaders[best_client_id], data_mean[0], data_std[0])
        print("test loss:", l, "\nMAE:", MAE, "RMSE:", RMSE, "MAPE:", MAPE * 100)

        metric_results['train_losses'][0].append(best_train_losses)
        metric_results['test_mae'].append(MAE)
        metric_results['test_rmse'].append(RMSE)
        metric_results['test_mape'].append(MAPE * 100)

    elif model_name == "SimVP":
        method = 'SimVP'
        # model
        spatio_kernel_enc = 3
        spatio_kernel_dec = 3
        model_type = "gsta"
        hid_S = 32
        hid_T = 256
        N_T = 8
        N_S = 2
        # training
        lr = 1e-3
        batch_size = 16
        drop_path = 0.1
        sched = 'cosine'
        warmup_epoch = 5
        in_shape = (12, 1, 10, 10)

        best_model = SimVP(in_shape, n_pred, hid_S, hid_T, N_S, N_T, model_type=model_type, mlp_ratio=8., drop=0.0,
                           drop_path=drop_path, spatio_kernel_enc=spatio_kernel_enc,
                           spatio_kernel_dec=spatio_kernel_dec, act_inplace=True).to(device)

        best_model.load_state_dict(torch.load(best_model_path))

        l = evaluate_model(best_model, loss, testloaders[best_client_id])
        # indexing first client data mean and std since test_iter is referenced as first client at data loading stage for simplification purposes
        MAE, RMSE, MAPE = evaluate_metric(best_model, testloaders[best_client_id], data_mean[0], data_std[0])
        print("test loss:", l, "\nMAE:", MAE, "RMSE:", RMSE, "MAPE:", MAPE * 100)

        metric_results['train_losses'][0].append(best_train_losses)
        metric_results['test_mae'].append(MAE)
        metric_results['test_rmse'].append(RMSE)
        metric_results['test_mape'].append(MAPE * 100)


    elif model_name == "TAU":
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
        in_shape = (12, 1, 10, 10)

        best_model = SimVP(in_shape, n_pred, hid_S, hid_T, N_S, N_T, model_type=model_type, mlp_ratio=8., drop=0.0,
                           drop_path=drop_path, spatio_kernel_enc=spatio_kernel_enc,
                           spatio_kernel_dec=spatio_kernel_dec, act_inplace=True).to(device)

        best_model.load_state_dict(torch.load(best_model_path))

        l = evaluate_model(best_model, loss, testloaders[best_client_id])
        # indexing first client data mean and std since test_iter is referenced as first client at data loading stage for simplification purposes
        MAE, RMSE, MAPE = evaluate_metric(best_model, testloaders[best_client_id], data_mean[0], data_std[0])
        print("test loss:", l, "\nMAE:", MAE, "RMSE:", RMSE, "MAPE:", MAPE * 100)

        metric_results['train_losses'][0].append(best_train_losses)
        metric_results['test_mae'].append(MAE)
        metric_results['test_rmse'].append(RMSE)
        metric_results['test_mape'].append(MAPE * 100)


    elif model_name == "GATAU":

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
        in_shape = (12, 1, 10, 10)

        best_model = GATAU(in_shape, n_pred, G, hid_S, hid_T, N_S, N_T, model_type=model_type, mlp_ratio=8., drop=0.0,
                           drop_path=drop_path, spatio_kernel_enc=spatio_kernel_enc,
                           spatio_kernel_dec=spatio_kernel_dec, act_inplace=True).to(device)

        best_model.load_state_dict(torch.load(best_model_path))

        l = evaluate_model(best_model, loss, testloaders[best_client_id])
        # indexing first client data mean and std since test_iter is referenced as first client at data loading stage for simplification purposes
        MAE, RMSE, MAPE = evaluate_metric(best_model, testloaders[best_client_id], data_mean[0], data_std[0])
        print("test loss:", l, "\nMAE:", MAE, "RMSE:", RMSE, "MAPE:", MAPE * 100)

        metric_results['train_losses'][0].append(best_train_losses)
        metric_results['test_mae'].append(MAE)
        metric_results['test_rmse'].append(RMSE)
        metric_results['test_mape'].append(MAPE * 100)


    print("metric_results: ", metric_results)
    df = pd.DataFrame.from_dict(metric_results)

    print(df)
    file_name = 'results.csv'
    row_contents = [city, num_clients, model_name, "Non-FL", metric_results['test_mae'][0], metric_results['test_mape'][0]]

    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(row_contents)