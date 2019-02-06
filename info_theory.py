# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:25:16 2018

@author: Anusha Lalitha
"""

import numpy as np

def entropyber(p):
    
    entropy = -p*np.log2(p)-(1-p)*np.log2(1-p)
    
    return entropy


def kldivber(p,q):
    
    kldivber = p*np.log2(p/q)+(1-p)*np.log2((1-p)/(1-q))
    
    return kldivber


def capacity_bsc(p):
    
    capacity = 1-entropyber(p)
    
    return capacity

def c1_bsc(p):
    
    c1 = kldivber(p, 1-p)
    
    return c1