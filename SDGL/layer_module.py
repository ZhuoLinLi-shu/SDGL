import torch
from torch import nn
import torch.nn.functional as F


class fc_layer(nn.Module):
    def __init__(self, in_channels, out_channels, need_layer_norm):
        super(fc_layer, self).__init__()
        self.linear_w = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_uniform_(self.linear_w.data, gain=1.414)

        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=[1, 1], bias=True)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.need_layer_norm = need_layer_norm

    def forward(self, input):
        '''
        input = batch_size, in_channels, nodes, time_step
        output = batch_size, out_channels, nodes, time_step
        '''
        if self.need_layer_norm:
            result = F.leaky_relu(torch.einsum('bani,io->bano ', [input.transpose(1, -1), self.linear_w]))\
                     # + self.layer_norm(self.linear(input).transpose(1, -1))
        else:
            result = F.leaky_relu(torch.einsum('bani,io->bano ', [input.transpose(1, -1), self.linear_w])) \
                     # + self.linear(input).transpose(1, -1)
        return result.transpose(1, -1)


class gatedFusion_1(nn.Module):
    def __init__(self, dim, device):
        super(gatedFusion_1, self).__init__()
        self.device = device
        self.dim = dim
        self.w = nn.Linear(in_features=dim, out_features=dim)
        self.t = nn.Parameter(torch.zeros(size=(self.dim, self.dim)))
        nn.init.xavier_uniform_(self.t.data, gain=1.414)
        self.norm = nn.LayerNorm(dim)
        self.re_norm = nn.LayerNorm(dim)

        self.w_r = nn.Linear(in_features=dim, out_features=dim)
        self.u_r = nn.Linear(in_features=dim, out_features=dim)

        self.w_h = nn.Linear(in_features=dim, out_features=dim)
        self.w_u = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, batch_size, nodevec, time_node):

        if batch_size == 1 and len(time_node.shape) < 3:
            time_node = time_node.unsqueeze(0)

        nodevec = self.norm(nodevec)
        node_res = self.w(nodevec) + nodevec
        # node_res = batch_size, nodes, dim
        node_res = node_res.unsqueeze(0).repeat(batch_size, 1, 1)

        time_res = time_node + torch.einsum('bnd, dd->bnd', [time_node, self.t])

        # z = batch_size, nodes, dim
        z = torch.sigmoid(node_res + time_res)
        r = torch.sigmoid(self.w_r(time_node) + self.u_r(nodevec).unsqueeze(0).repeat(batch_size, 1, 1))
        h = torch.tanh(self.w_h(time_node) + r * (self.w_u(nodevec).unsqueeze(0).repeat(batch_size, 1, 1)))
        res = torch.add(z * nodevec, torch.mul(torch.ones(z.size()).to(self.device) - z, h))

        return res