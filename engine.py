import numpy as np
import torch.optim as optim
from model import *
import util
class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        # lookback > receptive_field(13) 이면 time 차원이 1보다 큼 → 마지막 step만 사용
        output = output.transpose(1,3)[:, -1:, :, :]   # (B, 1, N, out_dim)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, np.nan)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,np.nan).item()
        rmse = util.masked_rmse(predict,real,np.nan).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        with torch.no_grad():
            input = nn.functional.pad(input,(1,0,0,0))
            output = self.model(input)
            output = output.transpose(1,3)[:, -1:, :, :]   # (B, 1, N, out_dim)
            real = torch.unsqueeze(real_val,dim=1)
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real, np.nan)
            mape = util.masked_mape(predict,real,np.nan).item()
            rmse = util.masked_rmse(predict,real,np.nan).item()
        return loss.item(),mape,rmse
