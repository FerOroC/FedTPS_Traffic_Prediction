from math import sin, cos, sqrt, atan2, radians, asin
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


def q_xt_x0(x0, t, alpha_bar):
    # get mean and var of xt given x0
    mean = gather(alpha_bar, t) ** 0.5 * x0
    var = 1 - gather(alpha_bar, t)
    # sample xt from q(xt | x0)
    eps = torch.randn_like(x0).to(x0.device)
    xt = mean + (var ** 0.5) * eps
    return xt, eps  # also returns noise


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a


def p_xt(xt, noise, t, next_t, beta, eta=0):
    at = compute_alpha(beta.cuda(), t.long())
    at_next = compute_alpha(beta, next_t.long())
    x0_t = (xt - noise * (1 - at).sqrt()) / at.sqrt()
    c1 = (eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    eps = torch.randn(xt.shape, device=xt.device)
    xt_next = at_next.sqrt() * x0_t + c1 * eps + c2 * noise
    return xt_next


def load_datasets(config, city, num_clients: int):
    inflow = np.load(f'data/{city}/{num_clients}_client/inflow_main.npy')
    inflow = inflow.reshape(inflow.shape[0], inflow.shape[1], 10, 10)
    inflow = torch.from_numpy(inflow).float()

    conditional = np.load(f'data/{city}/{num_clients}_client/inflow_conditional.npy')
    conditional = torch.from_numpy(conditional).float()
    conditional = conditional[:,:,:4]           #torch.cat((conditional[:,:,:2], conditional[:,:,3:5]), dim=-1)

    dataset = TensorDataset(inflow, conditional)

    trainloaders = []
    valloaders = []

    for inflow_set, conditional_set in dataset:
        len_train = int(inflow_set.shape[0] * 0.8)
        inflow_set_train = inflow_set[:len_train]
        conditional_set_train = conditional_set[:len_train]

        dataset_train = TensorDataset(inflow_set_train, conditional_set_train)

        trainloaders.append(DataLoader(dataset_train,
                                       batch_size=config.training.batch_size,
                                       shuffle=True))

        valloaders=None
    return trainloaders, valloaders