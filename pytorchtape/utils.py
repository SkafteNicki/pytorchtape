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
    
#%%
def accuracy_topk(output, target, topk=(5,10)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res