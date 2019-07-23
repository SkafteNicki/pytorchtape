# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:53:02 2019

@author: nsde
"""

import tensorflow as tf
tf.enable_eager_execution()

from data_utils.fluorescence_protein_serializer import deserialize_fluorescence_sequence as _deserialize_fluorescence_sequence
from data_utils.proteinnet_serializer import deserialize_proteinnet_sequence as _deserialize_proteinnet_sequence
from data_utils.remote_homology_serializer import deserialize_remote_homology_sequence as _deserialize_remote_homology_sequence
from data_utils.secondary_structure_protein_serializer import deserialize_secondary_structure as _deserialize_secondary_structure
from data_utils.stability_serializer import deserialize_stability_sequence as _deserialize_stability_sequence
from data_utils.pfam_protein_serializer import deserialize_pfam_sequence as _deserialize_pfam_sequence

import numpy as np
import os
import torch

#%%
_deserie_funcs = {'fluorescence':        _deserialize_fluorescence_sequence,
                  'proteinnet':          _deserialize_proteinnet_sequence,
                  'remotehomology':      _deserialize_remote_homology_sequence,
                  'secondarystructure':  _deserialize_secondary_structure,
                  'stability':           _deserialize_stability_sequence,
                  'pfam':                _deserialize_pfam_sequence}

#%%
class TFrecordToTorch(object):
    def __init__(self, recordfile, deserie_func, batch_size = 100, shuffle=True, pad_and_stack=True):
        # Set variables
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.pad_and_stack = pad_and_stack
        
        # Deserilization function
        self.deserie_func = deserie_func
        
        # Load tfrecord files
        if len(recordfile)==1:
            self.tfrecord = tf.data.TFRecordDataset(recordfile[0])
        else:
            self.tfrecord = tf.data.TFRecordDataset(recordfile[0])
            for i in range(1, len(recordfile)):
                self.tfrecord = self.tfrecord.concatenate(tf.data.TFRecordDataset(recordfile[i]))
        
        # Get size of dataset
        for i,dd in enumerate(self.tfrecord):
            pass
        self.N = i
        
        # Get number of batches
        self.dataset = self.tfrecord.batch(self.batch_size)
        for i,d in enumerate(self.dataset):
            pass
        self.n_batch = i
        
    def _deserialize(self, batch) -> list:
        return list(map(self.deserie_func, batch))
    
    def _convert(self, batch_list):
        keys = [k for k in batch_list[0].keys()]
        np_dict = dict()
        for k in keys:
            np_dict[k] = [ ]
            
        for b in batch_list:
            for k,v in b.items():
                np_dict[k].append(np.asarray(v.numpy()))
        
        if self.pad_and_stack:
            for k in keys:
                if k in ['primary', 'secondary_structure', 'solvent_accessibility',
                         'evolutionary', 'valid_mask', 'asa_max', 'disorder',
                         'interface', 'phi', 'psi', 'rsa', 'ss3', 'ss8']:
                    np_dict[k] = self._pad_to_length(np_dict[k], np_dict['protein_length'])
                
                # Stack arrays
                if k != 'contact_map':
                    try:
                        np_dict[k] = np.stack(np_dict[k])
                    except:
                        raise ValueError(k)
               
            # Convert to torch tensor (if possible)
            for k in keys:
                try:
                    np_dict[k] = torch.tensor(np_dict[k])
                except:
                    pass
                
            return np_dict
        else:
            return np_dict
    
    def _pad_to_length(self, X, L, n_elem=None):
        max_l = max(L)
        #n_elem = n_elem if n_elem!=None else X[0].shape[1]
        n_elem = 1 if len(X[0].shape)==1 else X[0].shape[1]
        return [np.concatenate((x.reshape(-1,n_elem), 
                np.asarray([0]*n_elem*(max_l - l)).reshape(max_l-l,n_elem))).squeeze() 
                for x,l in zip(X,L)]
    
    def _shuffle(self) -> None:
        self.dataset = self.tfrecord.shuffle(self.N).batch(self.batch_size)
    
    def __len__(self) -> int:
        return self.N
    
    def __iter__(self):
        for b in self.dataset:
            unserialized = self._deserialize(b)
            yield self._convert(unserialized)
        if self.shuffle: # Shuffle data after iterating through it
            self._shuffle()
    
#%%         
class Dataset(object):
    def __init__(self, batch_size=100, shuffle=True, pad_and_stack=True):
        # Files to read from 
        files = os.listdir(self.folder)
        self.train_files = [self.folder + '/' + f for f in files if 'train' in f]
        self.val_files = [self.folder + '/' + f for f in files if 'valid' in f ]
        self.test_files = [self.folder + '/' + f for f in files if 
                           (f not in self.train_files and f not in self.val_files)]
    
        # Set variables
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pad_and_stack = pad_and_stack
        
        # Deserielization function
        self.deserie_func = _deserie_funcs[self.__class__.__name__[:-7].lower()]
        
    @property
    def train_set(self):
        return TFrecordToTorch(self.train_files, self.deserie_func, 
                               self.batch_size, self.shuffle, self.pad_and_stack)
    
    @property
    def val_set(self):
        return TFrecordToTorch(self.train_files, self.deserie_func, 
                               self.batch_size, False, self.pad_and_stack)
    
    @property
    def test_set(self):
        return TFrecordToTorch(self.train_files, self.deserie_func, 
                               self.batch_size, False, self.pad_and_stack)

#%%
class FluorescenceDataset(Dataset):
    folder = 'data/fluorescence'
    
#%%
class ProteinnetDataset(Dataset):
    folder = 'data/proteinnet'
    
#%%
class RemotehomologyDataset(Dataset):
    folder = 'data/remote_homology'

#%%
class SecondarystructureDataset(Dataset):
    folder = 'data/secondary_structure'

#%%
class StabilityDataset(Dataset):
    folder = 'data/stability'
    
#%%
class PfamDataset(Dataset):
    folder = 'data/pfam'

#%%
if __name__ == '__main__':
    dataset = ProteinnetDataset(batch_size=10, shuffle=True, pad_and_stack=False)
    train = dataset.train_set
    val = dataset.val_set
    test = dataset.test_set
    
    for i, dd in enumerate(train):
        print(i)
        if i > 10:
            break
        
    for i, dd in enumerate(val):
        print(i)
        if i > 10:
            break
        
    for i, dd in enumerate(test):
        print(i)
        if i > 10:
            break
