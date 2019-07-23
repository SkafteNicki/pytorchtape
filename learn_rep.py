# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:24:06 2019

@author: nsde
"""

import argparse

from pytorchtape.datasets import PfamDataset
from pytorchtape.embed_models import FullyConnectedVAE

#%% #TODO: replace with argparse
class args:
    batch_size = 100

#%%
if __name__ == "__main__":
    # Initialize dataset
    dataset = PfamDataset(batch_size = args.batch_size,
                          shuffle = True)
    train_set = dataset.train_set
    val_set = dataset.val_set

    # Initialize model    
    model = 
    