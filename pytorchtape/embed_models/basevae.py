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

#%%
class BaseVAE(nn.Module):
    ''' Base class for vae model. Subclass all models with this class.
    
    Only need is to implement the following 3 methods:
        @_init_enc_dec_funcs(self):
            Initialize all parameterized functions here
        @encoder(self, x)
            Encoder of the vae. x has shape [batch_size, length, n_vocab].
            Should return both z_mu, z_var with shape [batch_size, latent_size]
        @decoder(self, z)
            Decoder of the vae. z has shape [n_sample, batch_size, latent_size]
            Should return single x_mu with shape [n_sample, batch_size, length, n_voacb]
    
    Other methods that should not be implemented:
        @__init__
        @forward
        @sample
        @latent_rep
        @interpolate
        @count_parameters
    '''
    def _init_enc_dec_funcs(self):
        raise NotImplementedError
        
    def encoder(self, x, length=None):
        ''' x.shape = [batch_size, seq_length, n_vocab] '''
        raise NotImplementedError
    
    def decoder(self, z, x_one_hot=None, length=None):
        ''' z.shape = [n_sample, batch_size, latent_size] '''
        raise NotImplementedError
    
    def __init__(self, max_seq_len, latent_size, device='cpu'):
        self.max_seq_len = max_seq_len
        self.latent_size = latent_size
        self.device = torch.device(device)
        self.emb_f = lambda x: F.one_hot(x.long(), num_classes=n_vocab).float()
        self.prior = D.Independent(D.Normal(torch.zeros(1,latent_size,device=self.device),
                                            torch.ones(1,latent_size,device=self.device)), 1)
        
        # Initialize parameters
        super().__init__()
        self._init_enc_dec_funcs()
        
        # Move to gpu if device=='cuda'
        if torch.cuda.is_available() and device=='cuda':
            self.cuda()
        
    def forward(self, seq, length=None, n_sample=1):
        one_hot_seq = self.emb_f(seq)
        
        # Encoder step
        z_mu, z_std = self.encoder(one_hot_seq, length)
        q_dist = D.Independent(D.Normal(z_mu, z_std+1e-6), 1)
        z = q_dist.rsample((n_sample,))
        
        # Decoder step
        x_mu = self.decoder(z, one_hot_seq, length)
        p_dist = D.Categorical(logits=x_mu)
        
        return p_dist, q_dist, x_mu, z, z_mu, z_std
    
    def loss_f(self, target, p_dist, q_dist, beta=1.0, mask=pad_idx):
        # Calculate elbo
        logpx = p_dist.log_prob(target) # [N,B,L]
        logpx[:,target==mask]=0 # mask padding indices
        kl = D.kl_divergence(q_dist, self.prior)
        iw_elbo = logpx.sum(dim=-1) - beta*kl # [N,B,L] -> [N,B] 
        elbo = iw_elbo.logsumexp(dim=0) - np.log(logpx.shape[0])# [N,B] -> [B]
        
        # Calculate ete
        ete = (-logpx).sum(dim=-1).mean().exp()
        
        # Calculate accuracy
        logits = p_dist.logits
        preds = logits.mean(dim=0).argmax(dim=-1)
        acc = (target == preds.to(target.dtype))
        acc = acc[target != mask].float().mean()
        
        # Calculate perplexity
        probs = p_dist.probs
        logp = p_dist.logits
        perplexity = (-(probs * logp).sum(dim=-1)).exp()
        weights = torch.ones_like(perplexity)
        weights[:,target==mask] = 0
        perplexity = (perplexity * weights).sum() / (weights.sum() + 1e-10)
        
        metrics = {'loss': -elbo.mean(),
                   'logpx': logpx.mean(),
                   'kl': kl.mean(),
                   'acc': acc,
                   'etc': ete,
                   'perplexity': perplexity}
        
        return metrics
    
    def sample(self, N=1, z=None):
        with torch.no_grad():
            if z is None:
                z = self.prior.sample((N,))
            else:
                z = z.to(self.device)
            x_mu = self.decoder(z)
            return x_mu.argmax(dim=-1)
    
    def latent_rep(self, x):
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
    