import torch.optim as optim
from .model_timeSeries import *
from torch.optim import lr_scheduler
import torch


class trainer():
    def __init__(self, data, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, gcn_bool,
                 addaptadj, embed_dim, dropout_ingc=0.5, eta=1, gamma=0.0001, order=1, m=0.9, out_length=1,
                 layers=2, batch_size=64, dilation_exponential=2, ):

        self.model = SDGL_model(device, num_nodes, dropout, gcn_bool=gcn_bool, addaptadj=addaptadj,
                                in_dim=in_dim, seq_length=seq_length, residual_channels=nhid,
                                dilation_channels=nhid, skip_channels=nhid * 8, out_dim=out_length,
                                end_channels=nhid * 16, embed_dim=embed_dim, dropout_ingc=dropout_ingc,
                                eta=eta, gamma=gamma, m=m, layers=layers, batch_size=batch_size,
                                dilation_exponential=dilation_exponential)

        self.model.to(device)
        self.gc_order = order

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                                        patience=10, eps=0.00001, cooldown=20, verbose=True)

        self.loss = nn.L1Loss(size_average=False).to(device)
        self.loss_mse = nn.MSELoss(size_average=False).to(device)

        self.clip = 0.5
        self.loss_usual = nn.SmoothL1Loss()

    def train(self, input, real_val, data, pred_time_embed=None, iter=0):
        self.model.train()
        # input = 32, 1, 137, 168
        output, gl_loss, _ = self.model(input, pred_time_embed)
        output = output.squeeze()

        real = real_val
        scale = data.scale.expand(real.size(0), data.m)
        real, predict = real * scale, output * scale

        if gl_loss is None:
            loss = self.loss(predict, real)
        else:
            loss = self.loss(predict, real) + torch.mean(gl_loss) * self.gc_order

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def eval(self, input, real_val, data, pred_time_embed=None):
        self.model.eval()
        with torch.no_grad():
            output, _, _ = self.model(input, pred_time_embed)
        output = output.squeeze()
        real = real_val
        scale = data.scale.expand(real.size(0), data.m)
        real, predict = real * scale, output * scale
        loss_mse = self.loss_mse(predict, real)
        loss = self.loss(predict, real)
        samples = (output.size(0) * data.m)
        return loss.item(), loss_mse.item(), samples, output
