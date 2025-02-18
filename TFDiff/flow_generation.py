import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm.notebook import tqdm
from types import SimpleNamespace
from utils.TFDiff_UNet import *
from utils.config import args
from utils.utils import *
from torch.utils.data import DataLoader
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_clients", required=True, help="Number of clients in FL setting", type=int)
parser.add_argument("--city", required=True, help="City in FL setting", type=str)
parser.add_argument("--batch_size", required=True, help="Batch size", type=int)

user_args = vars(parser.parse_args())

num_clients = user_args['num_clients']
city = user_args['city']
batch_size = user_args['batch_size']

def load_npz_file(file_path):
    loaded_data = np.load(file_path)
    array_list = [loaded_data[key] for key in loaded_data.keys()]
    return array_list

if __name__ == '__main__':
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)

    config = SimpleNamespace(**temp)

    unet = Guide_UNet(config).cuda()

    # # load the model using .pt
    state_dict = torch.load(f"small/epoch300_model.pt")

    unet.load_state_dict(state_dict, strict=True)

    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start,
                            config.diffusion.beta_end, n_steps).cuda()
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    lr = 2e-4  # Explore this - might want it lower when training on the full dataset

    eta=0.0
    timesteps=100
    skip = n_steps // timesteps
    seq = range(0, n_steps, skip)

    # load head information for guide trajectory generation
    batchsize = batch_size
    conditional_full = np.load(f'data/{city}/{num_clients}_client/inflow_conditional.npy',
                   allow_pickle=True)
    conditional_full = torch.from_numpy(conditional_full).float()
    conditional_full = conditional_full[:, -batch_size:]
    print("Head shape: ", conditional_full.shape)

    extra_data = np.load(f'data/{city}/{num_clients}_client/extra_data.npy', allow_pickle=True)

    clients_gen_traj = []
    clients_gen_head = []

    for client in range(num_clients):
        # ## Chengdu Rescaling Own Data Processing Values
        print(extra_data)
        extra_data_ = extra_data.ravel()[0]
        print(extra_data_)
        hmean = extra_data_['mean conditional data']
        hstd = extra_data_['std conditional data']
        mean = extra_data_['mean main data']
        std = extra_data_['std main data']


        conditional_client = conditional_full[client]
        print("conditional client shape: ", conditional_client.shape)
        dataloader = DataLoader(conditional_client, batch_size=batchsize, shuffle=False, num_workers=4)

        Gen_traj = []
        Gen_head = []
        for i in (range(1)):
            head = next(iter(dataloader))
            tes = head[:,:].numpy()
            Gen_head.extend((tes*hstd+hmean))
            head = head.cuda()
            # Start with random noise
            x = torch.randn(batchsize, config.data.channels, config.data.traj_length).cuda()
            ims = []
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                with torch.no_grad():
                    pred_noise = unet(x, t, head)
                    # print(pred_noise.shape)
                    x = p_xt(x, pred_noise, t, next_t, beta, eta)
                    if i % 10 == 0:
                        ims.append(x.cpu().squeeze(0))
            trajs = ims[-1].cpu().numpy()
            # resample the trajectory length
            for j in range(batchsize):
                new_traj = trajs[j]
                #new_traj = trajs[j].T
                #new_traj = new_traj[:,::-1]
                new_traj = new_traj * std + mean
                Gen_traj.append(new_traj)
        clients_gen_traj.append(Gen_traj)
        clients_gen_head.append(Gen_head)

print("Shape of output synthetic data: ", np.array(clients_gen_traj).shape)
with open(f"gen_data/{city}/{num_clients}_clients/{batch_size}_main.pkl", "wb") as f:
    pickle.dump(np.array(clients_gen_traj), f)

with open(f"/home/fermino/Documents/FedTPS_journal/DiffTFP/gen_data/{city}/{num_clients}_clients/{batch_size}_head.pkl", "wb") as f:
    pickle.dump(np.array(clients_gen_head), f)
