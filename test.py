# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:49:02 2019

@author: nsde
"""

from pytorchtape import get_task, vocab
from torch import nn

class args:
    max_length = 500
    batch_size = 10

class Reshape(nn.Module):
    def __init__(self, *sizes):
        super().__init__()
        self.sizes = sizes
        
    def forward(self, x):
        return x.view(-1, *self.sizes)

class Model(nn.Module):
    # Stupid autoencoder, just for test purpose
    def __init__(self):
        super().__init__()
        self.latent_size = 256 # MUST BE DEFINED
        self.encoder = nn.Sequential(nn.Embedding(len(vocab), 30),
                                     nn.Flatten(),
                                     nn.Linear(args.max_length * 30, 500),
                                     nn.ReLU(),
                                     nn.Linear(500, 256))
                                     
        self.decoder = nn.Sequential(nn.Linear(256, 500),
                                     nn.ReLU(),
                                     nn.Linear(500, args.max_length * len(vocab)),
                                     Reshape(args.max_length, len(vocab)))
        
    def forward(self, seq):
        latent = self.encoder(seq.long())
        return self.decoder(latent)
    
    def embed(self, seq): # MUST BE DEFINED
        return self.encoder(seq.long())

if __name__ == '__main__':
    model = Model().cuda()
    
    task = get_task('remotehomology')(model, fix_embedding=True).cuda()
    train, val, test = task.get_data()
    
    for batch in train:
        loss, metrics = task.loss_func(batch.cuda())
        break