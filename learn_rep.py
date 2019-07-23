# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:24:06 2019

@author: nsde
"""

import argparse
import torch

from pytorchtape.datasets import PfamDataset
from pytorchtape.embed_models import FullyconnectedVAE

#%% #TODO: replace with argparse
class args:
    batch_size = 100
    learning_rate = 1e-3
    max_seq_len = 500
    n_epochs = 10
    latent_size = 32

#%%
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize dataset
    dataset = PfamDataset(batch_size = args.batch_size,
                          shuffle = True)
    train_set = dataset.train_set
    val_set = dataset.val_set

    # Initialize model    
    model = FullyconnectedVAE(max_seq_len = args.max_seq_len,
                              latent_size = args.latent_size,
                              device = device)
    
    # Initialize optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning
    for i in range(args.n_epochs):
        for j, batch in enumerate(train_set):
            optim.zero_grad()
            
            # Prepare data
            input = _prepare_batch(batch, device)
            
            # Pass
            output = model(input)
            
            # Calculate loss
            loss, metrics = model.loss_f(output)
            
            # Do backprop
            loss.backward()
            optim.step()
            
    model.save()    
        
    
    

    
    