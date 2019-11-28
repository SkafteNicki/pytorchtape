# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:26:15 2019

@author: nsde
"""

#%%
import torch

#%% for dict of torch tensors (and other stuff too), that have the normal cuda, 
# cpu, detach and numpy methods
class cudaDict(dict):
    def cuda(self):
        for k,v in self.items():
            if type(v) == torch.Tensor:
                self[k]=v.cuda()
        return self
    
    def cpu(self):
        for k,v in self.items():
            if type(v) == torch.Tensor:
                self[k]=v.cpu()
        return self
                
    def detach(self):
        for k,v in self.items():
            if type(v) == torch.Tensor:
                self[k]=v.detach()
        return self
                
    def numpy(self):
        for k,v in self.items():
            if type(v) == torch.Tensor:
                self[k]=v.numpy()
        return self