# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:24:06 2019

@author: nsde
"""

#%%
import argparse
import torch
import numpy as np
from tqdm import tqdm
import pickle as pkl
import os

from pytorchtape.datasets import PfamDataset
from pytorchtape.embed_models import get_model
from pytorchtape.trainers import RepLearner

#%%
def argparser():
    parser = argparse.ArgumentParser()
    
    data_settings = parser.add_argument_group('Data settings')
    data_settings.add_argument('--logdir', type=str, default='results', help='folder to store model weights + logs')
    data_settings.add_argument('--n_data', type=int, default=1e9, help='number of datapoints to use in each epoch')
    data_settings.add_argument('--max_seq_len', type=int, default=100, help='maximum size of sequences trained on')
    
    training_settings = parser.add_argument_group('Training settings')
    training_settings.add_argument('--n_epochs', type=int, default=5, help='number of epochs to run fit algorithm')
    training_settings.add_argument('--batch_size', type=int, default=128, help='batch size')
    training_settings.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    training_settings.add_argument('--warmup', type=int, default=10000, help='kl anneling, increase from 0 to 1 by 1/warmup for each iteration ')
    training_settings.add_argument('--n_sample', type=int, default=5, help='number of samples used in the importance sampling')
    
    model_settings = parser.add_argument_group('Model settings')
    model_settings.add_argument('--model', type=str, default='fullyconnected', help='model to use')
    model_settings.add_argument('--latent_size', type=int, default=32, help='size of the latent space')
    
    args = parser.parse_args()
    return args

#%%
if __name__ == "__main__":
    args = argparser()
    if not os.path.exists('emb_models'):
        os.mkdir('emb_models')
    args.logdir = 'emb_models/' + args.logdir 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize dataset
    dataset = PfamDataset(batch_size = args.batch_size,
                          shuffle = True)
    train_set = dataset.train_set
    val_set = dataset.val_set
    test_set = dataset.test_set

    # Initialize model    
    model = get_model(args.model)(max_seq_len = args.max_seq_len,
                                  warmup_iters = args.warmup,
                                  latent_size = args.latent_size,
                                  device = device)
    
    # Initialize trainer
    trainer = RepLearner(model, args.max_seq_len, args.logdir, device=device)
    
    # Fit
    stats = trainer.fit(train_set,
                        val_set = val_set,
                        n_epochs = args.n_epochs,
                        learning_rate = args.lr,
                        data_pr_epoch = args.n_data)
    
    # Save training stats
    with open(args.logdir+'/stats.pkl', 'wb') as f:
        pkl.dump(stats, f)
        
    # Save model
    model.save(args.logdir)
    
    # Finally evaluate on test set
    metrics = trainer.evaluate
    with open(args.logdir+'/test_res.pkl', 'wb') as f:
        pkl.dump(metrics, f)