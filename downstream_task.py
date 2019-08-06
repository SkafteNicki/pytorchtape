# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:37:05 2019

@author: nsde
"""

#%%
import argparse
import pickle as pkl
import torch

from pytorchtape.datasets import get_dataset
from pytorchtape.task_models import get_task_model
from pytorchtape.embed_models import get_embed_model
from pytorchtape.trainers import TaskLearner


#%%
def argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--task', type=str, default='stability', help='downstream task to train')
    parser.add_argument('--embed_model', type=str, default='', help='embedding model to use')
    parser.add_argument('--logdir', type=str, default='results', help='folder to results')
    
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    args = parser.parse_args()
    return args
    
#%%
if __name__ == '__main__':
    args = argparser()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load embedding model
    embed_args = pkl.load(open(args.embed_model + '/args.pkl', 'rb'))
    embed_model = get_embed_model(embed_args.model)(max_seq_len = embed_args.max_seq_len,
                                              warmup_iters = embed_args.warmup,
                                              latent_size = embed_args.latent_size,
                                              device = device) 
    embed_model.load(args.embed_model + '/model_params')
    
    # Initialize task model
    task_model = get_task_model(args.task)(d_in = embed_args.latent_size)
    
    # Load dataset to embed
    dataset = get_dataset(args.task)(batch_size = args.batch_size,
                                     shuffle = False)
    
    trainer = TaskLearner(task_model, embed_model, embed_args.max_seq_len, args.logdir, device=device)    
    
    # Fit
    stats = trainer.fit(dataset.train_set,
                        n_epochs = args.n_epochs,
                        learning_rate = args.lr)
    # Evaluate
    train_res = trainer.evaluate(dataset.train_set)
    val_res = trainer.evaluate(dataset.val_set)
    test_res = trainer.evaluate(dataset.test_set)
    print(train_res)
    print(val_res)
    print(test_res)
    with open(args.logdir + '/results.pkl', 'wb') as f:
        pkl.dumb(f, train_res)
        pkl.dumb(f, val_res)
        pkl.dumb(f, test_res)
