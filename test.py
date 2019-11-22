# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:49:02 2019

@author: nsde
"""

from pytorchtape import get_dataset

if __name__ == '__main__':
    # Initialize a dataset
    dataset = get_dataset('pfam')(
            batch_size = 10, shuffle = True, max_length = 500)
    
    # Get train, val and test splits
    train = dataset.train_set 
    val = dataset.val_set
    test = dataset.test_set
    
    # We can only go through each subset in a iterative way, either as
    for batch in train:
        print(batch)
        break
    
    # or
    iterator = iter(train)
    for i in range(train.n_batch):
        batch = next(iterator)
        print(batch)
        break
    
    # Batch is always a dict (regardless of dataset), and always has these two fields:
    # * batch['primary'] <- padded amino acid sequences
    # * batch['protein_length'] <- length of each sequence
    # Additionally, each dataset comes with its own specific keys.
    
    