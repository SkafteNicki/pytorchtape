# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:40:35 2019

@author: nsde
"""

import torch
from torch import nn

#%%
def get_task_model(name):
    d = {'remotehomology': RemotehomologyModel,
         'fluorescence': FluorescenceModel,
         'stability': StabilityModel}
    assert name in d, '''Task model not found, please choose between {0}'''.format(
            [k for k in d.keys()])
    return d[name]

#%%
def pearson(x,y):
    vx = x - x.mean()
    vy = y - y.mean()
    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

#%%
def accuracy(x,y):
    return (x == y.argmax()).mean()

#%%
class RemotehomologyModel(nn.Module):
    corresponding_dataset = 'remotehomology'
    def __init__(self, d_in):
        super().__init__()
        
        self.net = nn.Sequential(nn.Linear(d_in, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1195))
        
        self._loss = nn.CrossEntropyLoss(reduction='mean')
        self._metric = accuracy
        
         # Move to gpu if device=='cuda'
        if torch.cuda.is_available():
            self.cuda()
        
    def forward(self, embedding, target):
        pred = self.net(embedding)
        loss = self._loss(target, pred)
        metric = self._metric(target, pred)
        metrics = {'loss': loss,
                   'acc': metric}
        return loss, metrics
    
#%%
class FluorescenceModel(nn.Module):
    corresponding_dataset = 'fluorescence'
    def __init__(self, d_in):
        super().__init__()
        
        self.net = nn.Sequential(nn.Linear(d_in, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))
        
        self._loss = nn.MSELoss(reduction='mean')
        self._metric = pearson
        
        # Move to gpu if device=='cuda'
        if torch.cuda.is_available():
            self.cuda()
    
    def forward(self, embedding, target):
        pred = self.net(embedding)
        loss = self._loss(target, pred)
        metric = self._metric(target, pred)
        metrics = {'loss': loss,
                   'pearson': metric}
        return loss, metrics
    
#%%
class StabilityModel(nn.Module):
    corresponding_dataset = 'stability'
    def __init__(self, d_in):
        super().__init__()
        
        self.net = nn.Sequential(nn.Linear(d_in, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))
        
        self._loss = nn.MSELoss(reduction='mean')
        self._metric = pearson
        
        # Move to gpu if device=='cuda'
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, embedding, target):
        pred = self.net(embedding)
        loss = self._loss(target, pred)
        metric = self._metric(target, pred)
        metrics = {'loss': loss,
                   'pearson': metric}
        return loss, metrics