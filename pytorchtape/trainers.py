# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:48:22 2019

@author: nsde
"""

#%%
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .utils import TrainingLogger


#%%
class RepLearner(object):
    def __init__(self, model, max_seq_len, logdir, device='cpu'):
        # Create logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        
        # Save input
        self.model = model
        self.max_seq_len = max_seq_len
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        
        # Set device
        self.device = torch.device(device)    

    def _pad(self, x):
        return np.concatenate((x, [0]*(self.max_seq_len - x.shape[0])))

    def prepare_batch(self, batch):
        seq = list(map(self._pad, batch['primary']))
        seq = [s for s in seq if s.shape[0] <= self.max_seq_len]
        seq = torch.tensor(np.stack(seq), device=self.device)
        length = torch.tensor(np.stack(batch['protein_length']), device=self.device)
        return {'input': seq,
                'target': seq,
                'length': length}
    
    def fit(self, train_set, n_epochs=1, n_sample=1, learning_rate = 1e-2, 
            val_set = None, val_epoch=1, data_pr_epoch=1e9):
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Log of training statistics
        logger = TrainingLogger()

        # Main loop
        logger.train_start()
        for i in range(1, n_epochs+1):
            logger.epoch_start()
            
            progressBar = tqdm(total=train_set.N, unit='samples',
                               desc='Epoch {0}/{1}'.format(i, n_epochs))
            self.model.train()
            for j, batch in enumerate(train_set):
                logger.batch_start()
                it = i*train_set.n_batch+j # global index
                
                # Prepare data
                data = self.prepare_batch(batch)
                
                # Forward pass
                loss, metrics = self.model(data, n_sample)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log stats
                for key, val in metrics.items():
                    logger.log_stat('train/'+key, val.item())

                # Save to tensorboard
                logger.dumb_to_tensorboard(self.writer, global_step=it)    
                logger.batch_end()
                
                progressBar.update(train_set.batch_size)
                progressBar.set_postfix([(k,v.item()) for k,v in metrics.items()])
                if (j+1)*train_set.batch_size > data_pr_epoch:
                    break
            
            progressBar.close()
            
            self.model.eval()
            if val_set and i % val_epoch == 0:
                metrics = self.evaluate(val_set)
                
                # Log stats
                for key, val in metrics.items():
                    logger.log_stat('val/'+key, val.item())
                    
                # Save to tensorboard
                logger.dumb_to_tensorboard(self.writer, global_step=it, filt='val')
                
            logger.epoch_end()
        
        # End of training
        logger.train_end()
            
        # Return stats
        return logger
    
    def evaluate(self, val_set):
        # Calculate metrics
        accumulate_metrics = dict()
        progressBar = tqdm(total=val_set.N, desc='Validation', unit='samples')
        for j, batch in enumerate(val_set):
            data = self.prepare_batch(batch)
            loss, metrics = self.model(data, 1)
                        
            for key, val in metrics.items():
                if key in accumulate_metrics:
                    accumulate_metrics[key].append(val.item())
                else:
                    accumulate_metrics[key] = [val.item()]
            
            progressBar.update(val_set.batch_size)
        progressBar.close()
        
        # Mean over the dataset
        metrics = dict()
        for key, val in accumulate_metrics.items():
            metrics[key] = np.asarray(val).mean()
        return metrics
    
#%%
class TaskLearner(object):
    def __init__(self, task_model, embed_model, max_seq_len, logdir, device='cpu'):
        # Create logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
            
        # Save input
        self.task_model = task_model
        self.embed_model = embed_model
        self.max_seq_len = max_seq_len
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        
        # Set device
        self.device = torch.device(device)
    
    def _pad(self, x):
        return np.concatenate((x, [0]*(self.max_seq_len - x.shape[0])))
    
    def prepare_batch(self, batch):
        seq = list(map(self._pad, batch['primary']))
        seq = [s for s in seq if s.shape[0] <= self.max_seq_len]
        seq = torch.tensor(np.stack(seq), device=self.device)
        length = torch.tensor(np.stack(batch['protein_length']), device=self.device)
        
        if 'stability_score' in batch.keys():
            target = batch['stability_score']
            target = [t for t,l in zip(target,batch['protein_length']) if l <= self.max_seq_len]
            target = torch.tensor(np.stack(target), device=self.device)
        
        return {'input': seq,
                'target': target,
                'length': length}
    
    def fit(self, train_set, n_epochs=1, learning_rate = 1e-2, val_set = None,
            val_epoch=1):
        # Optimizer
        optimizer = torch.optim.Adam(self.task_model.parameters(), lr=learning_rate)
        
        # Log of training statistics
        logger = TrainingLogger()
        
        # Main loop
        logger.train_start()
        for i in range(1, n_epochs+1):
            logger.epoch_start()
            
            progressBar = tqdm(total=train_set.N, unit='samples',
                               desc='Epoch {0}/{1}'.format(i, n_epochs))
            self.task_model.train()
            for j, batch in enumerate(train_set):
                logger.batch_start()
                it = i*train_set.n_batch+j # global index
                
                # Prepare data
                data = self.prepare_batch(batch)
                
                # Embed data
                emb = self.embed_model.embedding(data['input'])
                
                # Forward pass of task model
                loss, metrics = self.task_model(emb, data['target'])
                
                # Backward pass of task model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log stats
                for key, val in metrics.items():
                    logger.log_stat('train/'+key, val.item())

                # Save to tensorboard
                logger.dumb_to_tensorboard(self.writer, global_step=it)    
                logger.batch_end()
                
                progressBar.update(train_set.batch_size)
                progressBar.set_postfix([(k,v.item()) for k,v in metrics.items()])
            
            progressBar.close()
    
    def evaluate(self, val_set):
        # Calculate metrics
        accumulate_metrics = dict()
        progressBar = tqdm(total=val_set.N, desc='Validation', unit='samples')
        for j, batch in enumerate(val_set):
            data = self.prepare_batch(batch)
            emb = self.embed_model.embedding(data['input'])
            loss, metrics = self.task_model(emb, data['target'])
            for key, val in metrics.items():
                if key in accumulate_metrics:
                    accumulate_metrics[key].append(val.item())
                else:
                    accumulate_metrics[key] = [val.item()]
            
            progressBar.update(val_set.batch_size)
        progressBar.close()
        
        # Mean over the dataset
        metrics = dict()
        for key, val in accumulate_metrics.items():
            metrics[key] = np.asarray(val).mean()
        return metrics