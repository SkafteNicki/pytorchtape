# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:53:02 2019

@author: nsde
"""

import tensorflow as tf
<<<<<<< HEAD
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
#tf.enable_eager_execution(config=tf.ConfigProto(gpu_options=gpu_options))
=======
try:
    tf.enable_eager_execution() # if tf-1
except:
    pass # else tf-2 do nothing
>>>>>>> 0a9c7445756288db64d06d15a0f92674569d97b7

from .data_utils.fluorescence_protein_serializer import deserialize_fluorescence_sequence as _deserialize_fluorescence_sequence
from .data_utils.proteinnet_serializer import deserialize_proteinnet_sequence as _deserialize_proteinnet_sequence
from .data_utils.remote_homology_serializer import deserialize_remote_homology_sequence as _deserialize_remote_homology_sequence
from .data_utils.secondary_structure_protein_serializer import deserialize_secondary_structure as _deserialize_secondary_structure
from .data_utils.stability_serializer import deserialize_stability_sequence as _deserialize_stability_sequence
from .data_utils.pfam_protein_serializer import deserialize_pfam_sequence as _deserialize_pfam_sequence
from .data_utils.vocabs import PFAM_VOCAB
import numpy as np
import os
import torch
from Bio import SeqIO

#%%
_deserie_funcs = {'fluorescence':        _deserialize_fluorescence_sequence,
                  'proteinnet':          _deserialize_proteinnet_sequence,
                  'remotehomology':      _deserialize_remote_homology_sequence,
                  'secondarystructure':  _deserialize_secondary_structure,
                  'stability':           _deserialize_stability_sequence,
                  'pfam':                _deserialize_pfam_sequence}

#%%
class TFrecordToTorch(object):
    def __init__(self, recordfile, deserie_func, data_size, 
                 batch_size = 100, shuffle=True, pad_and_stack=True):
        # Set variables
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.pad_and_stack = pad_and_stack
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
    def __init__(self, batch_size=100, shuffle=True, pad_and_stack=False):
        # Files to read from 
        self.files = os.listdir(self.folder)
        self.train_files = [self.folder + '/' + f for f in self.files if 'train' in f]
        self.val_files = [self.folder + '/' + f for f in self.files if 'valid' in f ]
        self.test_files = [self.folder + '/' + f for f in self.files if 
                           (f not in self.train_files and f not in self.val_files)]
    
        # Set variables
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pad_and_stack = pad_and_stack
        
        # Deserielization function
        self.deserie_func = _deserie_funcs[self.__class__.__name__[:-7].lower()]
    
    @property
    def train_set(self):
        return TFrecordToTorch(self.train_files, self.deserie_func, self._train_N,
                               self.batch_size, self.shuffle, self.pad_and_stack)
    
    @property
    def val_set(self):
        return TFrecordToTorch(self.val_files, self.deserie_func, self._val_N,
                               self.batch_size, False, self.pad_and_stack)
    
    @property
    def test_set(self):
        return TFrecordToTorch(self.test_files, self.deserie_func, self._test_N,
                               self.batch_size, False, self.pad_and_stack)

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
                               self.deserie_func, self._test_N, 
                               self.batch_size, False, self.pad_and_stack)

#%%
class subset(object):
    def __init__(self, bs, sh, ids, seq, lengths, labels):
        self.bs = bs
        self.sh = sh
        self.ids = ids
        self.seq = seq
        self.lengths = lengths
        self.labels = labels
        
        self.N = len(self.ids)
        self.nb = int(np.ceil(self.N / self.bs))
    
    def __len__(self):
        return self.N
    
    def __iter__(self):
        np_dict = {}
        for i in range(self.nb):
            np_dict['id'] = self.ids[i*self.bs:(i+1)*self.bs]
            np_dict['primary'] = self.seq[i*self.bs:(i+1)*self.bs]
            np_dict['protein_length'] = self.lengths[i*self.bs:(i+1)*self.bs]
            np_dict['secondary_structure'] = self.labels[i*self.bs:(i+1)*self.bs]
            yield np_dict
        if self.sh: # shuffle
            perm = np.random.permutation(self.N)
            temp1, temp2, temp3, temp4 = [ ], [ ], [ ], [ ]
            for i in range(self.N):
                temp1.append(self.ids[perm[i]])
                temp2.append(self.seq[perm[i]])
                temp3.append(self.lengths[perm[i]])
                temp4.append(self.labels[perm[i]])
            self.ids = temp1
            self.seq = temp2
            self.lengths = temp3
            self.labels = temp4

#%%
class ScopeDataset(object):
    def __init__(self, batch_size=100, shuffle=True, pad_and_stack=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Load sequences
        ids, lab, seq, lengths = [ ], [ ], [ ], [ ]
        fasta_sequences = SeqIO.parse(open('data/scope/astral-scopedom-seqres-gd-sel-gs-bib-95-2.07.fa'),'fasta')
        for fasta in fasta_sequences:
            ids.append(np.array(fasta.id))
            lab.append(fasta.description[8])
            seq.append(np.array([PFAM_VOCAB[s] for s in str(fasta.seq).upper()]))
            lengths.append(np.array(len(seq[-1])))
        
        N = len(ids)
        self._train_N = int(N*0.8) # 80% train
        self._val_N = int(N*0.1) # 10% val
        self._test_N = N - self._train_N - self._val_N # 10% test
        
        # Convert labels
        label_conv = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6}
        temp = [ ]
        for l in lab:
            temp.append(np.array(label_conv[l]))
        lab = temp
        
        perm = np.random.permutation(N)
        self.train_id, self.val_id, self.test_id = [ ], [ ], [ ]
        self.train_seq, self.val_seq, self.test_seq = [ ], [ ], [ ]
        self.train_lengths, self.val_lengths, self.test_lengths = [ ], [ ], [ ]        
        self.train_labels, self.val_labels, self.test_labels = [ ], [ ], [ ]
        for i in range(N):
            if i < self._train_N:
                self.train_id.append(ids[perm[i]])
                self.train_seq.append(seq[perm[i]])
                self.train_lengths.append(lengths[perm[i]])
                self.train_labels.append(lab[perm[i]])
            elif self._train_N <= i < self._train_N + self._val_N:
                self.val_id.append(ids[perm[i]])
                self.val_seq.append(seq[perm[i]])
                self.val_lengths.append(lengths[perm[i]])
                self.val_labels.append(lab[perm[i]])
            else:
                self.test_id.append(ids[perm[i]])
                self.test_seq.append(seq[perm[i]])
                self.test_lengths.append(lengths[perm[i]])
                self.test_labels.append(lab[perm[i]])
    
    @property
    def train_set(self):
        return subset(self.batch_size, self.shuffle, self.train_id, self.train_seq, 
                      self.train_lengths, self.train_labels)
                
    @property
    def val_set(self):
        return subset(self.batch_size, False, self.train_id, self.train_seq, 
                      self.train_lengths, self.train_labels)
    
    @property
    def test_set(self):
        return subset(self.batch_size, False, self.train_id, self.train_seq, 
                      self.train_lengths, self.train_labels)

#%%
def get_dataset(name):
    d = {'fluorescence': FluorescenceDataset,
         'proteinnet': ProteinnetDataset,
         'remotehomology': RemotehomologyDataset,
         'secondarystructure': SecondarystructureDataset,
         'stability': StabilityDataset,
         'pfam': PfamDataset,
         'scope': ScopeDataset}
    assert name in d, '''Unknown dataset, choose from {0}'''.format(
            [k for k in d.keys()])
    return d[name]

#%%
if __name__ == '__main__':
    for classe in [FluorescenceDataset,
                   #ProteinnetDataset,
                   #RemotehomologyDataset,
                   #SecondarystructureDataset,
                   #StabilityDataset,
                   #PfamDataset
                   ]:
        print(classe)
        c = classe()
        #s1 = c.train_set
        #s2 = c.val_set
        #s3 = c.test_set

