# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:24:16 2019

@author: nsde
"""

#%%
import argparse

from pytorchtape.datasets import get_dataset
from pytorchtape.embed_models import get_model

#%%
def argparser()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--embed_model', type=str, default='', help='logdir of embedding model to use')
    parser.add_argument('--dataset', type=str, default='', help='dataset to embed')
    
    args = parser.parse_args()
    return args
#%%
if __name__ == "__main__":
    # Load model
    model = get_model(args.embed_model)()    
    
    # Load dataset to embed
    dataset = get_dataset(args.dataset)
    
    # Go through all of it
    logdir = 'embeddings/' + args.embed_model '/' + args.dataset
    names = ['train', 'val', 'test']
    for n, ds in zip(names, [dataset.train_set,
                             dataset.val_set,
                             dataset.test_set]):
        # Create logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        
        with open(logdir + '_' + n + '.pkl', 'wb') as f:
            for i, batch in enumerate(ds):
                # Prepare data
                seq = list(map(self.pad, batch['primary']))
                seq = [s for s in seq if s.shape[0] <= self.max_seq_len]
                seq = torch.tensor(np.stack(seq), device=device)
            
                # Get embeddings
                emb = model.embedding(seq)
            
                pkl.dumb(emb, file)
                
                
            
        
        