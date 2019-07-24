# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:24:16 2019

@author: nsde
"""

if __name__ == "__main__":
    # Load model
    
    # Load dataset to embed
    
    # Go through all of it
    for ds in [dataset.train_set,
               dataset.val_set,
               dataset.test_set]
        # Create logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        
        for i, batch in enumerate(ds):
            # Prepare data
            seq = list(map(self.pad, batch['primary']))
            seq = [s for s in seq if s.shape[0] <= self.max_seq_len]
            seq = torch.tensor(np.stack(seq), device=device)
            
            # Get embeddings
            emb = model.embedding(seq)
            
            pkl.dumb(emb, file)
            
        
        