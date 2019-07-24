# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:28:20 2019

@author: nsde
"""

#%%
import numpy as np
import time

#%%
class TrainingLogger(object):
    def __init__(self):
        self.stats = []
        self.iter_count = 0
        self.timings_T, self.timings_E, self.timings_B = [], [], []
        self._during_batch = False
        
    def log_stat(self, tag, val):
        if tag in self.stats[-1]:
            self.stats[-1][tag].append(val)
        else:
            self.stats[-1][tag] = [val]
    
    def train_start(self):
        self.timings_T.append(time.time())
        
    def train_end(self):
        self.timings_T.append(time.time())
        
        # Convert timings
        self.timings_T = np.array(self.timings_T)
        self.timings_E = np.array(self.timings_E)
        self.timings_B = np.array(self.timings_B)
        
        # Always comes in pairs if logged correctly
        self.timings_T = self.timings_T[::2] - self.timings_T[1::2]
        self.timings_E = self.timings_E[::2] - self.timings_E[1::2]
        self.timings_B = self.timings_B[::2] - self.timings_B[1::2]
        
    def epoch_start(self):
        self.timings_E.append(time.time())
        
    def epoch_end(self):
        self.timings_E.append(time.time())
        
    def batch_start(self):
        self.timings_B.append(time.time())
        self.stats.append(dict())
        self._during_batch = True
        
    def batch_end(self):
        self.iter_count += 1
        self.timings_B.append(time.time())
        self._during_batch = False
        
    def get_mean(self, tag, epoch=-1):
        return np.mean(self.stats[epoch][tag])
    
    def get_sum(self, tag, epoch=-1):
        return np.sum(self.stats[epoch][tag])
    
    def dumb_to_tensorboard(self, summarywriter, global_step=0):
        for key, val in self.stats[-1].items():
            if self._during_batch: # write last element
                summarywriter.add_scalar(key, val[-1], global_step=global_step)
            else:
                summarywriter.add_scalar(key, np.mean(val), global_step=global_step)