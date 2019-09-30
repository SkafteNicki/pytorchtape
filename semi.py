# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:38:35 2019

@author: nsde
"""

#%%
#import torch
#from torch import nn
#from pytorchtape.datasets import PfamDataset, StabilityDataset
#from .data_utils.vocabs import n_vocab, pad_idx
#
##%%
#class args:
#    max_seq_len = 500
#    
#
##%%
#class semi(nn.Module):
#    def __init__(self):
#        enc = nn.Sequential(nn.Linear(n_vocab, pad_idx))
#        
#        self.enc_mu = 
#        
#        self.enc_std = 
#        
#        self.classifier = nn.Sequential(nn.Linear())
#        
#        self.dec = 
#    
#    def forward(self, x, y=None):
#        
#
##%%
#if __name__ == "__main__":
#    device = 'cuda' if torch.cuda.is_available() else 'cpu'
#    
#    clas
#    
#    
#    
#    
#    
#

from pytorchtape.datasets import StabilityDataset as Dataset1
from pytorchtape.datasets import ScopeDataset as Dataset2

d1 = Dataset1(batch_size=5)
d2 = Dataset2(batch_size=5)

train1 = d1.train_set
train2 = d2.train_set

for i, batch1 in enumerate(train1):
    if i == 0:
        break
    
for i, batch2 in enumerate(train2):
    if i == 0:
        break