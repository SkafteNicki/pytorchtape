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
        seq = list(map(self.pad, batch['primary']))
        seq = [s for s in seq if s.shape[0] <= self.max_seq_len]
        seq = torch.tensor(np.stack(seq), device=self.device)
        length = torch.tensor(np.stack(batch['protein_length']), device=self.device)
        return {'input': seq,
                'target': seq,
                'length': length}
    
    def fit(self, train_set, batch_size=100, n_epochs=1, n_sample=1,
            learning_rate = 1e-2, val_set = None, warmup=1, val_epoch=1):
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Log of training statistics
        logger = TrainingLogger()

        # Main loop
        logger.train_start()
        for i in range(1, n_epochs+1):
            logger.epoch_start()
            
            progressBar = tqdm(total=train_set.n_batch, desc='Epoch {0}/{1}'.format(i, n_epochs))
            beta = np.minimum(1, i/warmup) # anneling weight for KL term
            self.model.train()
            for j, batch in enumerate(train_set):
                logger.batch_start()
                it = i*train_set.n_batch+j # global index
                
                # Prepare data
                data = self.prepare_batch(batch)
                
                # Forward pass
                loss, metrics = self.model(data, n_sample)
                
                # Compute loss, and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log stats
                for key, val in metrics.items():
                    logger.log_stat('train/'+key, val.item())

                # Save to tensorboard
                logger.dumb_to_tensorboard(self.writer, global_step=it)    
                logger.batch_end()
                
                progressBar.update()
                progressBar.set_postfix([(k,v.item()) for k,v in metrics.items()])
            
            progressBar.close()
            
            # Evaluate on validation set if present                
            self.model.eval()
            if val_set and i % val_epoch == 0: # evaluation on validation set    
                cm_r = np.zeros((n_vocab-5, n_vocab-5))
                for batch in val_loader:
                    # Get current data
                    data = batch['data'].to(self.device)
                    target = batch['target'].to(self.device)
                    length = batch['length'].to(self.device)
                    
                    p_dist, q_dist, x_mu, z, z_mu, z_std = self.model(data, length, n_sample)
                    metrics = self.model.loss_f(target, p_dist, q_dist, beta)
                    
                    for key, val in metrics.items():
                        logger.log_stat('val/'+key, val.item())
                    logger.log_stat('val/z_std', z_std.mean().item())
                    
                    pred = x_mu.mean(dim=0).argmax(dim=-1)
                    cm_r += conf_mat(target, pred)
                    
                logger.dumb_to_tensorboard(self.writer, global_step=it)
                self._save_recon(pred[:3], data[:3], length[:3], 'val/recon', global_step=it)
                self.writer.add_figure('val/conf_mat', plot_confusion_matrix(cm_r), global_step=it)
                self.writer.add_histogram('val/true_hist', data.flatten(),
                                          bins=list(range(n_vocab)), global_step=it)
                self.writer.add_histogram('val/pred_hist', pred.flatten(),
                                          bins=list(range(n_vocab)), global_step=it)
            logger.epoch_end()
        
        # End of training
        logger.train_end()
            
        # Save latent space
        self._save_embedding(train_loader, step=it, tag='train')
        if val_set:
            self._save_embedding(val_loader, step=it, tag='val')
        
        # Return stats
        return logger
    
    #%%
    def evaluate(self, test_set, batch_size=100):
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size = batch_size)
        # Calculate metrics on test set
        accumulate_metrics = dict()
        for j, batch in enumerate(test_loader):
            data = batch['data'].to(self.device)
            target = batch['target'].to(self.device)
            length = batch['length'].to(self.device)
            
            p_dist, q_dist, x_mu, z, z_mu, z_std = self.model(data, length, 1)
            metrics = self.model.loss_f(target, p_dist, q_dist, 1)
                        
            for key, val in metrics.items():
                if key in accumulate_metrics:
                    accumulate_metrics[key].append(val.item())
                else:
                    accumulate_metrics[key] = [val.item()]
        
        # Print res
        stats = dict()
        for key, val in accumulate_metrics.items():
            stats[key] = np.asarray(val).mean()
        print(stats)
    
    #%%
    def _save_recon(self, preds, target, lengths, tag, global_step=0):
        text = ''
        for p,t,l in zip(preds, target, lengths):
            pred_seq = int_seq_to_aa(p[:l].cpu().numpy())
            true_seq = int_seq_to_aa(t[:l].cpu().numpy())
            text += pred_seq + '  \n  ' + true_seq
            text += '  \n \n  '
        self.writer.add_text(tag, text, global_step=global_step)
    
    #%%
    def _save_embedding(self, dataloader, tag, step=0):
        with torch.no_grad():
            zs = [ ]
            
            for i, batch in enumerate(dataloader):
                if 'labels' in batch and i == 0:
                    haslabels = True
                    labels = [ ]
                elif 'labels' not in batch:
                    labels = None
                # Get current data
                data = batch['data'].to(self.device)
                length = batch['length'].to(self.device)
                
                _, _, _, _, z_mu, _ = self.model(data, length, 1)
                zs.append(z_mu.squeeze().cpu())
                
                if haslabels:
                    labels.append(batch['labels'])
                
            zs = torch.cat(zs)
            if zs.shape[1] < 3: # make sure that its 3D features
                zs = torch.cat([zs, torch.ones(zs.shape[0],3-zs.shape[1])], dim=1)
            if haslabels:
                labels = np.concatenate(labels)
                        
            self.writer.add_embedding(zs, metadata=labels, global_step=step, tag=tag)
            