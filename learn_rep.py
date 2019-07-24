# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:24:06 2019

@author: nsde
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

from pytorchtape.datasets import PfamDataset
from pytorchtape.embed_models import FullyconnectedVAE

#%% #TODO: replace with argparse
class args:
    batch_size = 256
    learning_rate = 1e-3
    max_seq_len = 500
    n_epochs = 10
    latent_size = 256
    n_sample = 10

#%%
def prepare_batch(batch, max_l, device):
    def pad(x):
        return np.concatenate((x, [0]*(max_l - x.shape[0])))
    seq = list(map(pad, batch['primary']))
    seq = [s for s in seq if s.shape[0] <= max_l]
    seq = torch.tensor(np.stack(seq), device=device)
    length = torch.tensor(np.stack(batch['protein_length']), device=device)
    return {'input': seq,
            'target': seq,
            'length': length}

#%%
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize dataset
    dataset = PfamDataset(batch_size = args.batch_size,
                          shuffle = True)
    train_set = dataset.train_set
    val_set = dataset.val_set
    test_set = dataset.test_set

    # Initialize model    
    model = FullyconnectedVAE(max_seq_len = args.max_seq_len,
                              latent_size = args.latent_size,
                              device = device)
    
    # Initialize optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning
    for i in range(args.n_epochs):
        beta = np.minimum(1, i/(args.n_epochs/2))
        progressBar = tqdm(total=train_set.n_batch, desc='Epoch {0}/{1}'.format(i, args.n_epochs))
        for j, batch in enumerate(train_set):
            optim.zero_grad()
            
            # Prepare data
            data = prepare_batch(batch, args.max_seq_len, device)
            data['beta'] = beta
            
            # Pass forward
            loss, metrics = model(data, args.n_sample)
            
            # Do backprop
            loss.backward()
            optim.step()
            
            # Update
            progressBar.update()
            progressBar.set_postfix([(k,v.item()) for k,v in metrics.items()])
        
        for j, batch in enumerate(val_set):
            data = prepare_batch(batch, args.max_seq_len, device)
            data['beta'] = beta
            loss, metrics = model(data, args.n_sample)
        
    model.save()    
        
    
    

    
    