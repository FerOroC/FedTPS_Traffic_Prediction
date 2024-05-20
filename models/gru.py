import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, num_nodes, window):
        super(GRU, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_nodes = num_nodes
        self.window = window
        self.gru = nn.GRU(num_nodes*input_size, hidden_size, num_layers, batch_first=True)
        self.final_gru = nn.GRU(hidden_size, num_nodes, 1, batch_first=True)

    def forward(self, x):
        self.batch_size = x.shape[0]
        x = x.reshape(self.batch_size, self.window, self.num_nodes*self.input_size)
        out, h = self.gru(x)
        out, h = self.final_gru(out[:, -1, :])
        return out