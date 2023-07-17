import torch
from torch import nn
from .layer_module import *


class Graph_learn(nn.Module):
    def __init__(self, node_dim, heads, head_dim, nodes=207, eta=1,
                 gamma=0.001, dropout=0.5, n_clusters=5):
        super(Graph_learn, self).__init__()

        self.D = heads * head_dim  # node_dim #
        self.heads = heads
        self.dropout = dropout
        self.eta = eta
        self.gamma = gamma

        self.head_dim = head_dim
        self.node_dim = node_dim
        self.nodes = nodes

        self.query = fc_layer(in_channels=node_dim, out_channels=self.D, need_layer_norm=False)
        self.key = fc_layer(in_channels=node_dim, out_channels=self.D, need_layer_norm=False)
        self.value = fc_layer(in_channels=node_dim, out_channels=self.D, need_layer_norm=False)
        self.mlp = nn.Conv2d(in_channels=self.heads, out_channels=self.heads, kernel_size=(1, 1), bias=True)

        self.bn = nn.LayerNorm(node_dim)

        self.w = nn.Parameter(torch.zeros(size=(nodes, node_dim)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.attn_static = nn.LayerNorm(nodes)
        self.skip_norm = nn.LayerNorm(nodes)
        self.attn_norm = nn.LayerNorm(nodes)
        self.linear_norm = nn.LayerNorm(nodes)
        self.attn_linear = nn.Parameter(torch.zeros(size=(nodes, nodes)))
        nn.init.xavier_uniform_(self.attn_linear.data, gain=1.414)
        self.attn_linear_1 = nn.Parameter(torch.zeros(size=(nodes, nodes)))
        nn.init.xavier_uniform_(self.attn_linear_1.data, gain=1.414)
        self.static_inf_norm = nn.LayerNorm(nodes)
        self.attn_norm_1 = nn.LayerNorm(nodes)
        self.attn_norm_2 = nn.LayerNorm(nodes)

    def forward(self, nodevec_fusion, nodevec_s, node_input, nodevec_dy, batch_size=64):
        batch_size, nodes, node_dim = batch_size, self.nodes, self.node_dim
        node_orginal = nodevec_s
        # Static Graph Structure Learning
        adj_static = self.static_graph(node_orginal)

        nodevec_fusion = self.bn(nodevec_fusion)

        # Inductive bias
        static_graph_inf = self.static_inf_norm(torch.mm(nodevec_dy, nodevec_dy.transpose(1, 0)))

        # residual connection in Dynamic relationship construction
        nodevec1_1 = torch.einsum('bnd, nl -> bnl', nodevec_fusion, self.w) + nodevec_fusion
        skip_atten = torch.einsum('bnd,bdm->bnm', nodevec1_1, nodevec1_1.transpose(-1, -2))
        skip_atten = self.skip_norm(skip_atten)

        # Multi-Head Adjacent mechanism
        nodevec_fusion = nodevec_fusion.unsqueeze(1).transpose(1, -1)
        query = self.query(nodevec_fusion)
        key = self.key(nodevec_fusion)
        # value = self.value(nodevec_fusion)
        key = key.squeeze(-1).contiguous().view(batch_size, self.heads, self.head_dim, nodes)
        query = query.squeeze(-1).contiguous().view(batch_size, self.heads, self.head_dim, nodes).transpose(-1, -2)
        attention = torch.einsum('bhnd, bhdu-> bhnu', query, key)
        attention /= (self.head_dim ** 0.5)
        attention = F.dropout(attention, self.dropout, training=self.training)
        attention = self.mlp(attention) + attention
        adj_bf = self.attn_norm(torch.sum(attention, dim=1)) + skip_atten

        # feedforward neural network
        adj_af = F.relu(torch.einsum('bnm, ml->bnl', self.linear_norm(adj_bf), self.attn_linear))
        adj_af = torch.einsum('bnm, ml -> bnl', adj_af, self.attn_linear_1)

        # add & norm
        dy_adj_inf = self.attn_norm_1(adj_af + adj_bf)
        dy_adj_inf = F.dropout(dy_adj_inf, self.dropout, training=self.training)

        # add Inductive bias
        static_graph_inf = static_graph_inf.unsqueeze(0).repeat(batch_size, 1, 1)
        dy_adj = self.attn_norm_2(dy_adj_inf + static_graph_inf)

        # The final inferred dynamic graph structure
        adj_dynamic = F.softmax(F.relu(dy_adj), dim=2)
        adj_static = adj_static.unsqueeze(0).repeat(batch_size, 1, 1)

        # Graph Structure Learning Loss
        gl_loss = None
        if self.training:
            gl_loss = self.graph_loss_orginal(node_input, adj_static, self.eta, self.gamma)
        return adj_dynamic, adj_static, node_orginal, gl_loss,

    def static_graph(self, nodevec):
        resolution_static = torch.mm(nodevec, nodevec.transpose(1, 0))
        resolution_static = F.softmax(F.relu(self.attn_static(resolution_static)), dim=1)
        return resolution_static

    def graph_loss_orginal(self, input, adj, eta=1, gamma=0.001):
        B, N, D = input.shape
        x_i = input.unsqueeze(2).expand(B, N, N, D)
        x_j = input.unsqueeze(1).expand(B, N, N, D)
        dist_loss = torch.pow(torch.norm(x_i - x_j, dim=3), 2) * adj
        dist_loss = torch.sum(dist_loss, dim=(1, 2))
        f_norm = torch.pow(torch.norm(adj, dim=(1, 2)), 2)
        gl_loss = dist_loss + gamma * f_norm
        return gl_loss


class graph_constructor(nn.Module):
    def __init__(self, nodes, dim, device, time_step, cout=16, heads=4, head_dim=8,
                 eta=1, gamma=0.0001, dropout=0.5, m=0.9, batch_size=64, in_dim=2, is_add1=True):
        super(graph_constructor, self).__init__()
        self.embed1 = nn.Embedding(nodes, dim)

        self.m = m
        self.embed2 = nn.Embedding(nodes, dim)
        for param in self.embed2.parameters():
            param.requires_grad = False
        for para_static, para_w in zip(self.embed2.parameters(), self.embed1.parameters()):
            para_static.data = para_w.data

        self.device = device
        self.nodes = nodes
        self.time_step = time_step
        if is_add1:
            time_length = time_step + 1
        else:
            time_length = time_step

        self.trans_Merge_line = nn.Conv2d(in_dim, dim, kernel_size=(1, time_length), bias=True) # cout
        self.gate_Fusion_1 = gatedFusion_1(dim, device)

        self.graph_learn = Graph_learn(node_dim=dim, heads=heads, head_dim=head_dim, nodes=nodes,
                                       eta=eta, gamma=gamma, dropout=dropout)

        self.dim_to_channels = nn.Parameter(torch.zeros(size=(heads * head_dim, cout * time_step)))
        nn.init.xavier_uniform_(self.dim_to_channels.data, gain=1.414)
        self.skip_norm = nn.LayerNorm(time_step)
        self.time_norm = nn.LayerNorm(dim)

    def forward(self, input):
        batch_size, nodes, time_step = input.shape[0], self.nodes, self.time_step
        # Momentum update
        for para_dy, para_w in zip(self.embed2.parameters(), self.embed1.parameters()):
            para_dy.data = para_dy.data * self.m + para_w.data * (1 - self.m)

        node_input = input

        node_input = self.time_norm(self.trans_Merge_line(node_input).squeeze(-1).transpose(1, 2))
        idx = torch.arange(self.nodes).to(self.device)
        nodevec_static = self.embed1(idx)

        nodevec_dy = self.embed2(idx)
        # Information fusion module
        nodevec_fusion = self.gate_Fusion_1(batch_size, nodevec_static, node_input) + nodevec_static
        # graph learning module (static and dynamic graph)
        adj = self.graph_learn(nodevec_fusion, nodevec_static, node_input, nodevec_dy, batch_size)
        return adj


