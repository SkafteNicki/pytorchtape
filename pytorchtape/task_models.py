# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:40:35 2019

@author: nsde
"""

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
class BasetaskModel(nn.Module):
    def forward(self, embedding, target):
        pred = self.net(embedding)
        loss = self._loss(target, pred)
        return loss, pred

#%%
class RemotehomologyModel(BasetaskModel):
    def __init__(self, d_in):
        super().__init__()
        
        self.net = nn.Sequential(nn.Linear(d_in, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1195))
        
        self._loss = nn.CrossEntropyLoss(reduction='mean')
    
#%%
class FluorescenceModel(BasetaskModel):
    def __init__(self, d_in):
        super().__init__()
        
        self.net = nn.Sequential(nn.Linear(d_in, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))
        
        self._loss = nn.MSELoss(reduction='mean')
    
#%%
class StabilityModel(BasetaskModel):
    def __init__(self, d_in):
        super().__init__()
        
        self.net = nn.Sequential(nn.Linear(d_in, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))
        
        self._loss = nn.MSELoss(reduction='mean')
