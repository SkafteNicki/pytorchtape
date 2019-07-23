# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:37:19 2019

@author: nsde
"""

import torch
from torch import nn

from .basevae import BaseVAE
from ..data_utils.vocabs import n_vocab

#%%
class FullyconnectedVAE(BaseVAE):
    def _init_enc_dec_funcs(self):
        self.enc_mu = nn.Sequential(nn.Linear(n_vocab*self.max_seq_len, 256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128, self.latent_size))
        self.enc_std = nn.Sequential(nn.Linear(n_vocab*self.max_seq_len, 256),
                                     nn.LeakyReLU(),
                                     nn.Linear(256, 128),
                                     nn.LeakyReLU(),
                                     nn.Linear(128, self.latent_size),
                                     nn.Softplus())
        self.dec_mu = nn.Sequential(nn.Linear(self.latent_size, 128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128, 256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, n_vocab*self.max_seq_len))

    def encoder(self, x, length=None):
        x = x.reshape(x.shape[0], -1)
        z_mu = self.enc_mu(x)
        z_std = self.enc_std(x)
        return z_mu, z_std
        
    def decoder(self, z, x_one_hot=None, length=None):
        x_mu = self.dec_mu(z)
        x_mu = x_mu.reshape(z.shape[0], z.shape[1], self.max_seq_len, n_vocab)
        return x_mu.log_softmax(dim=-1)