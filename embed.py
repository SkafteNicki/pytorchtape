# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:24:16 2019

@author: nsde
"""

#%%
import argparse
import pickle as pkl
import torch
import os
import numpy as np

from pytorchtape.datasets import get_dataset
from pytorchtape.embed_models import get_embed_model

#%%
def argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--embed_model', type=str, default='', help='logdir of embedding model to use')
    parser.add_argument('--dataset', type=str, default='', help='dataset to embed')
    
    args = parser.parse_args()
    return args

#%%
if __name__ == "__main__":
    # Get arguments
    args = argparser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get embed arguments, model ect.
    embed_args = pkl.load(open(args.embed_model + '/args.pkl', 'rb'))
    model = get_embed_model(embed_args.model)(max_seq_len = embed_args.max_seq_len,
                                              warmup_iters = embed_args.warmup,
                                              latent_size = embed_args.latent_size,
                                              device = device) 
    model.load(args.embed_model + '/model_params')
    
    # Crete embedding dir
    logdir = 'embeddings/' + args.embed_model.split('/')[-1] + '_' + args.dataset
    os.makedirs(logdir, exist_ok=True)
    
    # Load dataset to embed
    dataset = get_dataset(args.dataset)(batch_size = embed_args.batch_size,
                                        shuffle = False)
    
    def pad(x):
        return np.concatenate((x, [0]*(embed_args.max_seq_len - x.shape[0])))
    
    # Go through all of it
    names = ['train', 'val', 'test']
    for n, ds in zip(names, [dataset.train_set,
                             dataset.val_set,
                             dataset.test_set]):
        with open(logdir + '/' + n + '.pkl', 'wb') as f:
            for i, batch in enumerate(ds):
                # Prepare data
                seq = list(map(pad, batch['primary']))
                seq = [s for s in seq if s.shape[0] <= embed_args.max_seq_len]
                seq = torch.tensor(np.stack(seq), device=device)
            
                # Get embeddings
                emb = model.embedding(seq)
                
                # TODO: include some dataset info to catch the too long that are sorted away
                pkl.dump(emb, f)
    
    
    
    