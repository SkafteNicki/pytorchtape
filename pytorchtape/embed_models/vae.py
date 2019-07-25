# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:36:26 2019

@author: nsde
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D
import numpy as np

from ..data_utils.vocabs import n_vocab, pad_idx
from ..layers import BatchFlatten, BatchReshape

#%%
class BaseVAE(nn.Module):
    def _init_enc_dec_funcs(self):
        raise NotImplementedError
        
    def encoder(self, x):
        ''' x.shape = [batch_size, seq_length, n_vocab] '''
        raise NotImplementedError
    
    def decoder(self, z):
        ''' z.shape = [n_sample, batch_size, latent_size] '''
        raise NotImplementedError
    
    def __init__(self, latent_size, max_seq_len, warmup_iters, device='cpu'):
        self.max_seq_len = max_seq_len
        self.latent_size = latent_size
        self.device = torch.device(device)
        self.emb_f = lambda x: F.one_hot(x.long(), num_classes=n_vocab).float()
        self.prior = D.Independent(D.Normal(torch.zeros(1,latent_size,device=self.device),
                                            torch.ones(1,latent_size,device=self.device)), 1)
        self.warmup_iters = warmup_iters
        self.beta = 0
        
        # Initialize parameters
        super().__init__()
        self._init_enc_dec_funcs()
        
        # Move to gpu if device=='cuda'
        if torch.cuda.is_available() and device=='cuda':
            self.cuda()
        
    def forward(self, data, n_sample=1):
        one_hot_seq = self.emb_f(data['input'])
        
        # Encoder step
        z_mu, z_std = self.encoder(one_hot_seq)
        q_dist = D.Independent(D.Normal(z_mu, z_std+1e-6), 1)
        z = q_dist.rsample((n_sample,))
        
        # Decoder step
        x_mu = self.decoder(z)
        p_dist = D.Categorical(logits=x_mu)
        
        # Calculate loss
        self.beta += 1.0/self.warmup_iters if self.training else 0 
        self.beta = np.minimum(1,self.beta)
        target = data['target']
        logpx = p_dist.log_prob(target) # [N,B,L]
        logpx[:,target==pad_idx]=0 # mask padding indices
        kl = D.kl_divergence(q_dist, self.prior)
        iw_elbo = logpx.sum(dim=-1) - self.beta*kl # [N,B,L] -> [N,B] 
        elbo = iw_elbo.logsumexp(dim=0) - np.log(logpx.shape[0])# [N,B] -> [B]
        
        # Calculate ete
        ece = (-logpx).mean().exp()
        
        # Calculate accuracy
        logits = p_dist.logits
        preds = logits.mean(dim=0).argmax(dim=-1)
        acc = (target == preds.to(target.dtype))
        acc = acc[target != pad_idx].float().mean()
        
        # Calculate perplexity
        probs = p_dist.probs
        logp = p_dist.logits
        perplexity = (-(probs * logp).sum(dim=-1)).exp()
        weights = torch.ones_like(perplexity)
        weights[:,target==pad_idx] = 0
        perplexity = (perplexity * weights).sum() / (weights.sum() + 1e-10)
        
        # Return loss and metrics
        metrics = {'loss': -elbo.mean(),
                   'logpx': logpx.mean(),
                   'kl': kl.mean(),
                   'acc': acc,
                   'ece': ece,
                   'perplexity': perplexity,
                   'z_std': z_std.mean()}
        
        return metrics['loss'], metrics
    
    def sample(self, N=1, z=None):
        with torch.no_grad():
            if z is None:
                z = self.prior.sample((N,))
            else:
                z = z.to(self.device)
            x_mu = self.decoder(z)
            return x_mu.argmax(dim=-1)
    
    def embedding(self, x):
        with torch.no_grad():
            x_onehot = self.emb_f(x)
            return self.encoder(x_onehot)[0]
    
    def interpolate(self, z1=None, z2=None, N=10, mode='linear'):
        if z1 is None:
            z1 = self.prior.sample()
        else:
            z1 = z1.to(self.device)
        if z2 is None:
            z2 = self.prior.sample()
        else:
            z2 = z2.to(self.device)
        assert z1.shape==[1,self.latent_size], \
            'Shape needs to be [1, self.latent_size] for z1'
        assert z2.shape==[1,self.latent_size], \
            'Shape needs to be [1, self.latent_size] for z2'
    
        if mode=='linear':
            w = torch.linspace(0,1,N,device=self.device).reshape(-1,1)
            z = (1-w)*z1 + w*z2
        elif mode=='gaussian':
            w = torch.linspace(0,1,N,device=self.device).reshape(-1,1)
            z = (1-w).sqrt()*z1 + w.sqrt()*z2
        elif mode=='riemannian':
            raise NotImplementedError
        else:
            raise ValueError('Unknown interpolation mode')
    
        x_mu = self.decoder(z)
        return x_mu.argmax(dim=-1)
    
    def count_parameters(self):
        c = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Total number of parameters:', c)
    
    def save_model(self, logdir):
        torch.save(self.state_dict(), logdir+'/model_params')
         
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        
#%%
class FullyconnectedVAE(BaseVAE):
    def _init_enc_dec_funcs(self):
        enc = nn.Sequential(nn.Linear(n_vocab*self.max_seq_len, 1024),
                            nn.LeakyReLU(),
                            nn.Linear(1024, 512),
                            nn.LeakyReLU())
        self.enc_mu = nn.Sequential(enc,
                                    nn.Linear(512, self.latent_size))
        self.enc_std = nn.Sequential(enc,
                                     nn.Linear(512, self.latent_size),
                                     nn.Softplus())
        self.dec_mu = nn.Sequential(nn.Linear(self.latent_size, 128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128, 256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, n_vocab*self.max_seq_len))

    def encoder(self, x):
        x = x.reshape(x.shape[0], -1)
        z_mu = self.enc_mu(x)
        z_std = self.enc_std(x)
        return z_mu, z_std
        
    def decoder(self, z):
        x_mu = self.dec_mu(z)
        x_mu = x_mu.reshape(z.shape[0], z.shape[1], self.max_seq_len, n_vocab)
        return x_mu.log_softmax(dim=-1)
    
#%%        
class ConvVAE(BaseVAE):
    def _init_enc_dec_funcs(self):
        enc = nn.Sequential(nn.Conv1d(n_vocab, 128, 3, stride=2),
                            nn.BatchNorm1d(128),
                            nn.LeakyReLU(),
                            nn.Conv1d(128, 64, 3, stride=2),
                            nn.BatchNorm1d(64),
                            nn.LeakyReLU())
        out_size = enc(torch.randint(0,n_vocab,(10,n_vocab,self.max_seq_length)).float()).shape

        self.enc_mu = nn.Sequential(enc,
                                    BatchFlatten(),
                                    nn.Linear(out_size[1]*out_size[2], self.latent_size))
        self.enc_std = nn.Sequential(enc,
                                     BatchFlatten(),
                                     nn.Linear(out_size[1]*out_size[2], self.latent_size),
                                     nn.Softplus())
        
        self.dec_mu = nn.Sequential(nn.Linear(self.latent_size, out_size[1]*out_size[2]),
                                    BatchReshape((out_size[1], out_size[2])),
                                    nn.ConvTranspose1d(64, 128, 3, stride=2),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(),
                                    nn.ConvTranspose1d(128, n_vocab, 3, stride=2),
                                    nn.BatchNorm1d(n_vocab),
                                    nn.LeakyReLU())
    # bs, 
    def encoder(self, x, length=None):
        x = x.permute(0,2,1)
        z_mu = self.enc_mu(x)
        z_std = self.enc_std(x)
        return z_mu, z_std
    
    def decoder(self, z, x_one_hot=None, length=None):
        z_r = z.reshape(-1,*z.shape[2:]) # merge n_sample and batch dim
        x_mu = self.dec_mu(z_r)
        x_mu = x_mu.reshape(z.shape[0], z.shape[1], *x_mu.shape[1:])
        return x_mu.transpose(perm=[0,1,3,2]).log_softmax(dim=-1)