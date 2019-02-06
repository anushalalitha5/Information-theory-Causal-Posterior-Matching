# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:25:16 2018

@author: Anusha Lalitha
"""

import numpy as np
import info_theory as it
import scipy.optimize


def sphere_packing_bound(p, rate, x0):
    
    const = 1 - rate
    fun = lambda x: const + x*np.log2(x) + (1-x)*np.log2(1-x)
    x_opt = scipy.optimize.fsolve(fun, x0)
    err_exp = it.kldivber(x_opt, p) 
    
    return err_exp

def random_coding(p, rate, x0):
    
    delta_cr = np.sqrt(p)/(np.sqrt(p) + np.sqrt(1-p))
    R_cr = 1 - it.entropyber(delta_cr)

    if(rate <= R_cr):
        err_exp = 1- 2*np.log2(np.sqrt(p) + np.sqrt(1-p))-rate
    elif (rate > R_cr):
        err_exp = sphere_packing_bound(p, rate, x0) 
        
    return err_exp


def burnashev_VL(p, rate):
    
    capacity = 1-it.entropyber(p)
    c1 = it.kldivber(p,1-p)
    
    err_exp = c1*(1-rate/capacity)
    
    return err_exp


def henderson_FL(p, rate, div, bound):
    
    q = 1-p
    end_pt = 2
    start_pt = end_pt/div
    beta_vec = np.linspace(start_pt, end_pt, div)
    sol_vec = np.zeros((div), dtype=float)
    dummy_vec = np.zeros((div), dtype=float)
    
    for j in range(div):
        beta = beta_vec[j]
        
        fun = lambda x: x*beta + np.log2(p*(2*p)**x+ q*(2*q)**x)
        xopt = scipy.optimize.fminbound(func=fun, x1=0, x2=10, disp = 0)
        fopt = fun(xopt)
        sol_vec[j] = (-fopt)- rate - beta
        if (sol_vec[j] >= -bound) and (sol_vec[j] <= bound):
            dummy_vec[j] = 1


    ind_vec = [x for x in range(len(dummy_vec)) if dummy_vec[x] == 1]
    
    if(len(ind_vec) > 0):
        err_exp = max(beta_vec[ind_vec])
    else: 
        err_exp = 0
        
    return err_exp


