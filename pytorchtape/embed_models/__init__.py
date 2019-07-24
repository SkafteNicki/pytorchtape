# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:32:46 2019

@author: nsde
"""

from .vae import FullyconnectedVAE

def get_model(name):
    d = {'fullyconnected': FullyconnectedVAE}
    return d[name]