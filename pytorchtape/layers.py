# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:07:17 2019

@author: nsde
"""

#%%
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.rnn import RNNBase, LSTMCell
from torch.nn import functional as F
import math

#%%
class BatchFlatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)
    
#%%
class BatchReshape(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes
    
    def forward(self, x):
        return x.reshape(-1, *self.sizes)
    
#%%
class mLSTM(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True,
                 dropout=0, bidirectional=False):
        super(mLSTM, self).__init__(
            mode='LSTM', input_size=input_size, hidden_size=hidden_size,
                 num_layers=num_layers, bias=bias, batch_first=True,
                 dropout=dropout, bidirectional=bidirectional)

        hidden_factor = 2 if bidirectional else 1

        w_im = torch.Tensor(hidden_factor*hidden_size, input_size)
        w_hm = torch.Tensor(hidden_factor*hidden_size, hidden_factor*hidden_size)
        b_im = torch.Tensor(hidden_factor*hidden_size)
        b_hm = torch.Tensor(hidden_factor*hidden_size)
        self.w_im = Parameter(w_im)
        self.b_im = Parameter(b_im)
        self.w_hm = Parameter(w_hm)
        self.b_hm = Parameter(b_hm)

        self.lstm_cell = LSTMCell(input_size, hidden_size, bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        n_batch, n_seq, n_feat = input.size()

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions,
                                n_batch, self.hidden_size, dtype=input.dtype,
                                device=input.device)
            hx = (zeros, zeros)

        hx, cx = hx
        steps = [cx.unsqueeze(1)]
        for seq in range(n_seq):
            mx = F.linear(input[:, seq, :], self.w_im, self.b_im) * F.linear(hx, self.w_hm, self.b_hm)
            hx = (mx, cx)
            hx, cx = self.lstm_cell(input[:, seq, :], hx)
            steps.append(cx.unsqueeze(1))

        return torch.cat(steps, dim=1)