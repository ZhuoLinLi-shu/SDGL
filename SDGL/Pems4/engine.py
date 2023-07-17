import torch.optim as optim
from .model_traffic import *
from torch.optim import lr_scheduler
import torch
from .lib.metrics import RMSE_torch, MAE_torch, MAPE_torch


class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, gcn_bool,
                 addaptadj, embed_dim, dropout_ingc=0.5, eta=1, gamma=0.0001, order=1, m=0.9,
                 layers=2, batch_size=64, dilation_exponential_=2,):

        self.model = SDGL_model(device, num_nodes, dropout, gcn_bool=gcn_bool, addaptadj=addaptadj,
                                in_dim=in_dim, out_dim=seq_length, residual_channels=nhid,
                                dilation_channels=nhid, skip_channels=nhid * 8,  # skip_channels,
                                end_channels=nhid * 16, embed_dim=embed_dim, dropout_ingc=dropout_ingc,  # end_channels
                                eta=eta, gamma=gamma, m=m, layers=layers, batch_size=batch_size,
                                dilation_exponential_=dilation_exponential_)

        self.model.to(device)
        self.gc_order = order
        nparams = sum([p.nelement() for p in self.model.parameters()])
        print(nparams)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.3,
                                                        patience=20, eps=0.00001, cooldown=30, verbose=True)

        self.loss = MAE_torch # masked_mae
        self.scaler = scaler
        self.clip = 5
        self.loss_usual = nn.SmoothL1Loss()

    def train(self, input, real_val, iter=0, pred_time_embed=None):
        self.model.train()

        input = nn.functional.pad(input, (1, 0, 0, 0))
        output, gl_loss, _ = self.model(input, pred_time_embed)
        output = output.transpose(1, 3)
        real = self.scaler.inverse_transform(real_val)
        predict = self.scaler.inverse_transform(output)
        """
        Sometimes using a custom loss (MAE_torch) can result in non convergence, and in this case, 
        replacing the loss with the official implementation of Pytorch
        """
        if gl_loss is None:
            loss = self.loss(predict, real, 0.0)
        else:
            loss = torch.mean(gl_loss) * self.gc_order + self.loss_usual(predict, real) # self.loss(predict, real, 0.0)

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()
        self.optimizer.zero_grad()

        mape = MAPE_torch(predict, real, 0.0).item()
        rmse = RMSE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, input, real_val, pred_time_embed=None):
        self.model.eval()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output, _, _ = self.model(input, pred_time_embed=pred_time_embed)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        # real = torch.unsqueeze(real_val, dim=1)
        real = self.scaler.inverse_transform(real_val)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = MAPE_torch(predict, real, 0.0).item() # masked_mape
        rmse = RMSE_torch(predict, real, 0.0).item() # masked_rmse
        return loss.item(), mape, rmse
