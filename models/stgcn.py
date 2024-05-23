"""
Reference to https://github.com/FelixOpolka/STGCN-PyTorch
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.conv import ChebConv

# x input shape required: [batch_size, channel/feature_size, window_size, number_nodes]

class TemporalConvLayer(nn.Module):
    """Temporal convolution layer.
    arguments
    ---------
    c_in : int
        The number of input channels (features)
    c_out : int
        The number of output channels (features)
    dia : int
        The dilation size
    """

    def __init__(self, c_in, c_out, kernel=2, dia=1):
        #Applies a 2D convolution over an input signal composed of several input planes, so captures some temporal correlation over sample window
        #dilation controls the spacing between the kernel points

        super(TemporalConvLayer, self).__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.conv = nn.Conv2d(
            c_in, 2*c_out, (kernel, 1), 1, dilation=dia, padding=(0, 0)
        )
        if self.c_in > self.c_out:
            self.conv_self = nn.Conv2d(c_in, c_out, (1,1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Applies GLU, Gated Linear Unit activation function, with residual connection - as shown in STGCN paper. 
        b, _, T, n = list(x.size())
        if self.c_in > self.c_out:
            x_self = self.conv_self(x)
        elif self.c_in < self.c_out:
            x_self = torch.cat([x, torch.zeros([b, self.c_out - self.c_in, T, n]).to(x)], dim=1)
        else:
            x_self = x
        conv_x = self.conv(x)
        # get the timesteps dim 
        _, _, T_new, _ = list(conv_x.size())
        x_self = x_self[:, :, -T_new:, :]
        P = conv_x[:, :self.c_out, :, :]
        Q = conv_x[:, -self.c_out:, :, :]
        gated_conv = torch.mul((P + x_self), self.sigmoid(Q))
        return gated_conv


class SpatioConvLayer(nn.Module):
    """Spatial Graph Conv Layer.
    arguments
    ---------
    c_in : int
        The number of input channels (features)
    c_out : int
        The number of output channels (features)
    graph : DGL.graph
        graph to be used for graph conv 
    """
    def __init__(self, c_in, c_out, graph):  
        super(SpatioConvLayer, self).__init__()
        self.g = graph
        self.gc = GraphConv(c_in, c_out, activation=F.relu)

        self.gc.reset_parameters()

    def forward(self, x):
        x = x.transpose(0, 3)
        x = x.transpose(1, 3)
        output = self.gc(self.g, x)
        output = output.transpose(1, 3)
        output = output.transpose(0, 3)
        return output



class OutputLayer(nn.Module):
    """Output layer.
    arguments
    ---------
    c : int
        The number of input/output channels (features)
    T : int
        kernel size
    n : int
        number of nodes
    """
    #Final output layer has two Temporal Convolution Layers, a normalisation layer inbetween, and one fully connected layer.
    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()
        self.tconv1 = nn.Conv2d(c, c, (T, 1), 1, dilation = 1, padding = (0,0))
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation = 1, padding = (0,0))
        self.fc = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        output = self.fc(x_t2)
        return output


class STGCN(nn.Module):
    def __init__(
        self, blocks, window, kt, n, graph, device, droprate, control_str="TSTNDTSTND"
    ):
        super(STGCN, self).__init__()
        # blocks = [T1_c_in, T1_c_out, S1_c_in_out, T2_c_out, N1_c_in,
        #           T3_c_in, T3_c_out, S2_c_in_out, T4_c_out, N2_c_in]
        #blocks = [1, 32, 32, 64, 64,
        #          64, 32, 32, 128, 128]
        self.control_str = control_str  # model structure controller
        self.num_layers = len(control_str)
        self.layers = nn.ModuleList([])
        self.droprate = droprate
        self.kt = kt
        self.window = window
        cnt = 0
        temporal_layers = 0 
        diapower = 0
        for i in range(self.num_layers):
            i_layer = control_str[i]
            if i_layer == "T":  # Temporal Layer
                self.layers.append(TemporalConvLayer(blocks[cnt], blocks[cnt + 1], kt))
                cnt += 2
                temporal_layers += 1
            if i_layer == "S":  # Spatio Layer
                self.layers.append(SpatioConvLayer(blocks[cnt], blocks[cnt], graph))
            if i_layer == "N":  # Norm Layer
                self.layers.append(nn.LayerNorm([n, blocks[cnt]]))
            if i_layer == "D": # Dropout Layer
                self.layers.append(nn.Dropout(p=self.droprate))
                cnt += 1
        #Find T (kernel size through simulation, assuming 2 on first pass)
        # T is window length - (kernel -1) * number of temporal layers
        self.output = OutputLayer(blocks[cnt-1], self.window - (self.kt - 1) * temporal_layers, n)
        for layer in self.layers:
            layer = layer.to(device)

    def forward(self, x):
        for i in range(self.num_layers):
            i_layer = self.control_str[i]
            if i_layer == "N":
                x = self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            elif i_layer == "S":
                x = self.layers[i](x)
            else:
                x = self.layers[i](x)
        return self.output(x)