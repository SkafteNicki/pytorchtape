# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:40:35 2019

@author: nsde
"""

import torch
from torch import nn

#%%
class RemotehomologyModel(nn.Module):
    def __init__(self, d_in):
        self.d_in = d_in
        
        self.net = nn.Sequential(nn.Linear(d_in, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1195))
        
        self._loss = nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, embedding):
        return self.net(embedding)
    
    def loss_f(self, target, preds):
        return self._loss(preds, target)
        
    
#%%
class FluorescenceModel(nn.Module):
    def __init__(self, d_in):
        self.d_in = d_in
        
        self.net = nn.Sequential(nn.Linear(d_in, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))
        
        self._loss = nn.MSELoss(reduction='mean')
        
    def forward(self, embedding):
        return self.net(embedding)
    
    def loss_f(self, target, preds):
        return self._loss(preds, target)
    
#%%
class StabilityModel(nn.Module):
    def __init__(self, d_in):
        self.d_in = d_in
        
        self.net = nn.Sequential(nn.Linear(d_in, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))
        
        self._loss = nn.MSELoss(reduction='mean')
        
    def forward(self, embedding):
        return self.net(embedding)
    
    def loss_f(self, target, preds):
        return self._loss(preds, target)