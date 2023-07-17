import torch.nn as nn
import torch.nn.functional as F
from ..graph_learn_module import graph_constructor
from SDGL.layer_mtgnn import *
import numpy as np


class dnconv(nn.Module):
    def __init__(self):
        super(dnconv, self).__init__()

    def forward(self, x, A):
        if len(A.size()) == 2:
            A = A.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = torch.einsum('nvw, ncwl->ncvl', A, x)
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn_module(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn_module, self).__init__()
        self.nconv = dnconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        x1 = self.nconv(x, support)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1, support)
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class SDGL_model(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, gcn_bool=True, addaptadj=True, seq_length=12,
                 in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=64, end_channels=128,
                 layers=2, embed_dim=10, dropout_ingc=0.5, eta=1, gamma=0.001,
                 m=0.9, batch_size=64, dilation_exponential_=1):
        super(SDGL_model, self).__init__()
        self.dropout = dropout

        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv_s = nn.ModuleList()
        self.gconv_d = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.nodes = num_nodes

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.seq_length = seq_length
        kernel_size = 7

        dilation_exponential = dilation_exponential_
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        rf_size_i = 1
        new_dilation = 1
        for j in range(1, layers + 1):
            if dilation_exponential > 1:
                # rf_size_j = 7, 19, 43, 91, 187
                rf_size_j = int(
                    rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
            else:
                rf_size_j = rf_size_i + j * (kernel_size - 1)

            self.filter_convs.append(
                dilated_inception(residual_channels, dilation_channels, dilation_factor=new_dilation))
            self.gate_convs.append(
                dilated_inception(residual_channels, dilation_channels, dilation_factor=new_dilation))

            self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=residual_channels,
                                                 kernel_size=(1, 1)))

            if self.seq_length > self.receptive_field:
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, self.seq_length - rf_size_j + 1)))
            else:
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, self.receptive_field - rf_size_j + 1)))

            if self.gcn_bool:
                self.gconv_s.append(gcn_module(dilation_channels, residual_channels, dropout, support_len=1, order=2))
                self.gconv_d.append(gcn_module(dilation_channels, residual_channels, dropout, support_len=1, order=2))

            if self.seq_length > self.receptive_field:
                self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                                           elementwise_affine=True))
            else:
                self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                                           elementwise_affine=True))
            new_dilation *= dilation_exponential
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length),
                                   bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1),
                                   bias=True)
        self.idx = torch.arange(self.nodes).to(device)

        self.graph_construct = graph_constructor(num_nodes, embed_dim, device, seq_length, eta=eta, in_dim=in_dim,
                                                 gamma=gamma, dropout=dropout_ingc, m=m, batch_size=batch_size)

    def forward(self, input, pred_time_embed=None):
        in_len = input.size(3)

        if in_len < self.receptive_field:
            x = nn.functional.pad(input, [self.receptive_field - in_len, 0, 0, 0])
        else:
            x = input

        static_adj, gl_loss, dy_adj = None, None, None

        if self.gcn_bool:
            adj_d, adj_s, node_embed, graph_loss = self.graph_construct(input)
            gl_loss = graph_loss
            static_adj = adj_s
            dy_adj = adj_d

        skip = self.skip0(F.dropout(x, self.dropout, training=self.training))
        x = self.start_conv(x)
        # WaveNet layers
        for i in range(self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)

            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)

            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            if self.gcn_bool:
                x_s = self.gconv_s[i](x, static_adj)
                x_d = self.gconv_d[i](x, dy_adj)
                x = x_s + x_d
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x, self.idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)

        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = x

        return x, gl_loss, None
