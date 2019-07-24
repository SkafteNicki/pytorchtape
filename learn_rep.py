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

from pytorchtape.datasets import PfamDataset
from pytorchtape.embed_models import get_model
from pytorchtape.trainers import RepLearner

#%%
def argparser():
    parser = argparse.ArgumentParser()
    
    data_settings = parser.add_argument_group('Data settings')
    data_settings.add_argument('--logdir', type=str, default='results')
    data_settings.add_argument('--max_seq_len', type=int, default=100)
    
    training_settings = parser.add_argument_group('Training settings')
    training_settings.add_argument('--n_epochs', type=int, default=5)
    training_settings.add_argument('--batch_size', type=int, default=128)
    training_settings.add_argument('--lr', type=float, default=1e-2)
    training_settings.add_argument('--warmup', type=int, default=1)
    training_settings.add_argument('--n_sample', type=int, default=5)
    
    model_settings = parser.add_argument_group('Model settings')
    model_settings.add_argument('--model', type=str, default='fullyconnected')
    model_settings.add_argument('--latent_size', type=int, default=32)
    
    args = parser.parse_args()
    return args

#%%
if __name__ == "__main__":
    args = argparser()
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
                        warmup)
    
    # Save training stats
    with open(args.logdir+'/stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
        
    # Save model
    model.save(logdir)
    
    # Finally evaluate on test set
    metrics = trainer.evaluate
    with open(args.logdir+'/test_res.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
#    # Initialize optimizer
#    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
#    
#    # Learning
#    for i in range(args.n_epochs):
#        beta = np.minimum(1, i/(args.n_epochs/2))
#        progressBar = tqdm(total=train_set.n_batch, desc='Epoch {0}/{1}'.format(i, args.n_epochs))
#        for j, batch in enumerate(train_set):
#            optim.zero_grad()
#            
#            # Prepare data
#            data = prepare_batch(batch, args.max_seq_len, device)
#            data['beta'] = beta
#            
#            # Pass forward
#            loss, metrics = model(data, args.n_sample)
#            
#            # Do backprop
#            loss.backward()
#            optim.step()
#            
#            # Update
#            progressBar.update()
#            progressBar.set_postfix([(k,v.item()) for k,v in metrics.items()])
#        
#        for j, batch in enumerate(val_set):
#            data = prepare_batch(batch, args.max_seq_len, device)
#            data['beta'] = beta
#            loss, metrics = model(data, args.n_sample)
#        
#    model.save()    
#        
#    
#    

    
    