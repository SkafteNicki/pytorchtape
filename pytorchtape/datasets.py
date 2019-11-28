# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:53:02 2019

@author: nsde
"""

import tensorflow as tf
try:
    tf.enable_eager_execution() # if tf-1
except:
    pass # else tf-2 do nothing

from .data_utils.fluorescence_protein_serializer import deserialize_fluorescence_sequence as _deserialize_fluorescence_sequence
from .data_utils.proteinnet_serializer import deserialize_proteinnet_sequence as _deserialize_proteinnet_sequence
from .data_utils.remote_homology_serializer import deserialize_remote_homology_sequence as _deserialize_remote_homology_sequence
from .data_utils.secondary_structure_protein_serializer import deserialize_secondary_structure as _deserialize_secondary_structure
from .data_utils.stability_serializer import deserialize_stability_sequence as _deserialize_stability_sequence
from .data_utils.pfam_protein_serializer import deserialize_pfam_sequence as _deserialize_pfam_sequence
from .data_utils.vocabs import PFAM_VOCAB as vocab
from .utils import cudaDict
import numpy as np
import os
import torch
from Bio import SeqIO
from torch.utils.data.dataloader import default_collate
from torch.utils.data import IterableDataset

#%%
_deserie_funcs = {'fluorescence':        _deserialize_fluorescence_sequence,
                  'proteinnet':          _deserialize_proteinnet_sequence,
                  'remotehomology':      _deserialize_remote_homology_sequence,
                  'secondarystructure':  _deserialize_secondary_structure,
                  'stability':           _deserialize_stability_sequence,
                  'pfam':                _deserialize_pfam_sequence}

#%%
class TFrecordToTorch(IterableDataset):
    def __init__(self, recordfile, deserie_func, data_size, 
                 batch_size = 100, shuffle=True, max_length=500):
        # Set variables
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.max_length = max_length
        self.N = data_size
        
        # Deserilization function
        self.deserie_func = deserie_func
        
        # Load tfrecord files
        if len(recordfile)==1:
            self.tfrecord = tf.data.TFRecordDataset(recordfile[0])
        else:
            self.tfrecord = tf.data.TFRecordDataset(recordfile[0])
            for i in range(1, len(recordfile)):
                self.tfrecord = self.tfrecord.concatenate(tf.data.TFRecordDataset(recordfile[i]))
        
        # Get number of batches
        self.dataset = self.tfrecord.batch(self.batch_size)
        self.n_batch = int(np.ceil(self.N / self.batch_size))
        
    def _deserialize(self, batch) -> list:
        return list(map(self.deserie_func, batch))
    
    def __len__(self) -> int:
        return self.N
    
    def __iter__(self):
        for b in self.dataset:
            unserialized = self._deserialize(b)
            yield self._collate_fn(unserialized)
            
        if self.shuffle: # Shuffle data after iterating through it
            self.dataset = self.tfrecord.shuffle(self.N).batch(self.batch_size)
            
    def _collate_fn(self, batch):
        out = cudaDict()
        for key in batch[0].keys():                     
            try:
                out[key] = default_collate(
                        [d[key].numpy() for d in batch if d['protein_length']<self.max_length])
            except RuntimeError:
                out[key] = torch.nn.utils.rnn.pad_sequence(
                        [torch.tensor(d[key].numpy().flatten()) for d in batch if d['protein_length']<self.max_length], 
                        padding_value=vocab['<PAD>'], batch_first=True)
        # Do extra padding for the protein sequences
        out['primary'] = torch.nn.functional.pad(out['primary'], (0,self.max_length-out['protein_length'].max()))
        return out
    
#%%         
class Dataset(object):
    def __init__(self, batch_size=100, shuffle=True, max_length=500):
        # Files to read from 
        self.folder = os.path.dirname(os.path.abspath(__file__)) + '/' + self.folder
        self.files = os.listdir(self.folder)
        
        self.train_files = [self.folder + '/' + f for f in self.files if 'train' in f]
        self.val_files = [self.folder + '/' + f for f in self.files if 'valid' in f ]
        self.test_files = [self.folder + '/' + f for f in self.files if 
                           (f not in self.train_files and f not in self.val_files)]
    
        # Set variables
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_length = max_length
        
        # Deserielization function
        self.deserie_func = _deserie_funcs[self.__class__.__name__[:-7].lower()]
    
    @property
    def train_set(self):
        return TFrecordToTorch(self.train_files, self.deserie_func, self._train_N,
                               self.batch_size, self.shuffle, self.max_length)
    
    @property
    def val_set(self):
        return TFrecordToTorch(self.val_files, self.deserie_func, self._val_N,
                               self.batch_size, False, self.max_length)
    
    @property
    def test_set(self):
        return TFrecordToTorch(self.test_files, self.deserie_func, self._test_N,
                               self.batch_size, False, self.max_length)

#%%
class FluorescenceDataset(Dataset):
    folder = 'data/fluorescence'
    _train_N = 21445
    _val_N = 5361
    _test_N = 54024
    
#%%
class ProteinnetDataset(Dataset):
    folder = 'data/proteinnet'
    _train_N = 25298
    _val_N = 223
    _test_N = 25562
    
#%%
class RemotehomologyDataset(Dataset):
    folder = 'data/remote_homology'
    _train_N = 12311
    _val_N = 735
    _test_N = 16291

#%%
class SecondarystructureDataset(Dataset):
    folder = 'data/secondary_structure'
    _train_N = 8677 
    _val_N = 2169
    _test_N = 11496

#%%
class StabilityDataset(Dataset):
    folder = 'data/stability'
    _train_N = 53613
    _val_N = 2511
    _test_N = 68976
    
#%%
class PfamDataset(Dataset):
    folder = 'data/pfam'
    _train_N = 32593667
    _val_N = 1715454
    _test_N = 44310
    
    @property
    def test_set(self):
        return TFrecordToTorch([self.folder + '/' + f for f in self.files if 'holdout' in f],
                               self.deserie_func, self._test_N, self.batch_size, 
                               False, self.max_length)

#%%
def get_dataset(name):
    d = {'fluorescence': FluorescenceDataset,
         'proteinnet': ProteinnetDataset,
         'remotehomology': RemotehomologyDataset,
         'secondarystructure': SecondarystructureDataset,
         'stability': StabilityDataset,
         'pfam': PfamDataset}
    assert name in d, '''Unknown dataset, choose from {0}'''.format(
            [k for k in d.keys()])
    return d[name]

#%%
if __name__ == '__main__':
    pass

