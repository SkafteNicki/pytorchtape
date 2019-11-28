# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:33:27 2019

@author: nsde
"""

from torch import nn
from .datasets import get_dataset
from scipy.stats import spearmanr

#%%
class Task(nn.Module):
    def __init__(self, embed_model, fix_embedding=True):
        super().__init__()
        self.embed_model = embed_model
        self.latent_size = self.embed_model.latent_size
        self.fix_embedding = fix_embedding
        if fix_embedding:
            for p in self.embed_model.parameters():
                p.requires_grad = False
        
        
    def forward(self, batch):
        embedding = self.embed_model.embed(batch['primary'])
        return self.predictor(embedding)
        
    def loss_func(self, batch):
        raise NotImplementedError
        
    def get_data(self, batch_size=10, max_length=500):
        raise NotImplementedError
        
#%%
class StabilityTask(Task):
    def __init__(self, embed_model, fix_embedding=True):
        super().__init__(embed_model, fix_embedding)
    
        self.predictor = nn.Sequential(nn.LayerNorm(self.latent_size),
                                       nn.Linear(self.latent_size, 500),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(500, 1))
    
        self.loss = nn.MSELoss()
    
    def loss_func(self, batch):
        prediction = self(batch)
        target = batch['stability_score']
        loss = self.loss(prediction, target)
        mae = (prediction-target).abs().mean()
        corr, _ = spearmanr(prediction.detach().cpu().numpy(), target.cpu().numpy())
        metrics = {'MSE': loss.item(), 'MAE': mae.item(), 'S_Corr': corr}
        return loss, metrics
    
    def get_data(self, batch_size=10, max_length=500):
        dataset = get_dataset('stability')(batch_size=batch_size)
        return dataset.train_set, dataset.val_set, dataset.test_set
    
    
#%%
class FluorescenceTask(Task):
    def __init__(self, embed_model, fix_embedding=True):
        super().__init__(embed_model, fix_embedding)
        
        self.predictor = nn.Sequential(nn.LayerNorm(self.latent_size),
                                       nn.Linear(self.latent_size, 500),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(500, 1))
    
        self.loss = nn.MSELoss()
        
    def loss_func(self, batch):
        prediction = self(batch)
        target = batch['stability_score']
        loss = self.loss(prediction, target)
        mae = (prediction-target).abs().mean()
        corr, _ = spearmanr(prediction.numpy(), target.numpy())
        metrics = {'MSE': loss, 'MAE': mae, 'S_Corr': corr}
        return loss, metrics
    
    def get_data(self, batch_size=10, max_length=500):
        dataset = get_dataset('fluorescence')(batch_size=batch_size)
        return dataset.train_set, dataset.val_set, dataset.test_set
    
#%%
def get_task(name):
    d = {'fluorescence': FluorescenceTask,
         #'proteinnet': ProteinnetDataset,
         #'remotehomology': RemotehomologyDataset,
         #'secondarystructure': SecondarystructureDataset,
         'stability': StabilityTask,
         #'pfam': PfamDataset
         }
    assert name in d, '''Unknown task, choose from {0}'''.format(
            [k for k in d.keys()])
    return d[name]