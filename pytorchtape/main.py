# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:08:50 2019

@author: nsde
"""
import torch

from datasets import PfamDataset, StabilityDataset
from task_models import StabilityModel

if __name__ == '__main__':
    print('Load datasets')
    dataset = PfamDataset(batch_size=256, shuffle=True, pad_and_stack=False)
    train_set = dataset.train_set
    val_set = dataset.val_set
    test_set = dataset.test_set    
    
    
    
    n_epochs_task = 10
    
    task_dataset = StabilityDataset(batch_size=256, shuffle=True, pad_and_stack=False)
    task_train = task_dataset.train_set
    task_val = task_dataset.val_set
    task_test = task_dataset.test_set
    
    task_model = StabilityModel(100)
    
    for i in range(n_epochs_task):
        for batch in task_train:
            sequence = torch.stack(batch['primary'])
            target = torch.stack(batch['stability'])
            
            embeddings = embed_model(sequence)
            
            pred = task_model(embeddings)
            loss = task_model.loss_f(target, pred)
            loss.backward()
            optim.step()
            
    g
            
    
            