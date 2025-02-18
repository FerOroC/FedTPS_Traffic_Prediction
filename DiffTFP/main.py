
import torch
import os
import torch.nn.functional as F
from utils.config import args
from utils.EMA import EMAHelper
from utils.TFDiff_UNet import *
from utils.utils import load_datasets, q_xt_x0
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--num_clients", required=True, help="Number of clients in FL setting", type=int)
parser.add_argument("--city", required=True, help="City in FL setting", type=str)
user_args = vars(parser.parse_args())

num_clients = user_args['num_clients']
city = user_args['city']
print("City being analysed: ", city)

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

# Load configuration
temp = {}
for k, v in args.items():
    temp[k] = SimpleNamespace(**v)
config = SimpleNamespace(**temp)

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)

def train(global_model, trainloader, config, num_local_epochs, device):
    local_model = copy.deepcopy(global_model)
    local_model = local_model.to(device)

    # Training params
    # Set up some parameters
    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end, n_steps).cuda()
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    lr = 2e-4  # Explore this - might want it lower when training on the full dataset

    losses = []  # Store losses for later plotting
    # optimizer
    optim = torch.optim.AdamW(local_model.parameters(), lr=lr)  # Optimizer

    # EMA
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(local_model)
    else:
        ema_helper = None
    # config.training.n_epochs = 1
    for epoch in range(num_local_epochs):
        for batch_id, (trainx, head) in enumerate(trainloader):
            x0 = trainx.cuda()
            head = head.cuda()
            t = torch.randint(low=0, high=n_steps,
                              size=(len(x0) // 2 + 1,)).cuda()
            t = torch.cat([t, n_steps - t - 1], dim=0)[:len(x0)]
            xt, noise = q_xt_x0(x0, t, alpha_bar)
            pred_noise = local_model(xt.float(), t, head)
            loss = F.mse_loss(noise.float(), pred_noise)
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            if config.model.ema:
                ema_helper.update(local_model)

        print("Average Epoch Loss: ", sum(losses) / len(losses))
    return local_model

def running_model_avg(current, next, scale):
    if current == None:
        current = next
        for key in current:
            current[key] = current[key] * scale
    else:
        for key in current:
            current[key] = current[key] + (next[key] * scale)
    return current

def fed_avg_experiment(global_model, num_clients, num_local_epochs, client_train_loader, max_rounds, device, city):
    for t in range(max_rounds):
        print("starting round {}".format(t))

        # choose clients
        print("Number of clients: ", num_clients)

        global_model.eval()
        global_model = global_model.to(device)
        running_avg = None

        for i in range(num_clients):
            # train local client
            print("round {}, starting client {}/{}, id: {}".format(t, i + 1, num_clients, i))
            local_model = train(global_model, client_train_loader[i], config, num_local_epochs, device)

            # add local model parameters to running average
            running_avg = running_model_avg(running_avg, local_model.state_dict(), 1 / num_clients)

        # set global model parameters for the next step
        global_model.load_state_dict(running_avg)

        if t % 300 == 0:
            save_model_path = f"{city}_models/{num_clients}_clients"
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            save_model_path = save_model_path + f'/epoch{t}_model.pt'
            torch.save(global_model.state_dict(), save_model_path)

    return global_model


# load datasets and adjacency matrix
trainloaders, valloaders = load_datasets(config, city, num_clients)

# model instantiation
unet = Guide_UNet(config).cuda()

trained_model = fed_avg_experiment(unet, num_clients=num_clients, num_local_epochs=1, client_train_loader=trainloaders,
                                   max_rounds=301, device=device, city=city)
