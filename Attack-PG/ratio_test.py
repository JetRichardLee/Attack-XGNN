# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:46:24 2024

@author: 31271
"""
import numpy as np


def compute_log_likelihood(degree, d_min=1):
    n=0
    sigma_log_div=0
    sigma_log=0
    for i in range(len(degree)):
        if degree[i]>=d_min:
            n+=1
            sigma_log_div+=np.log(degree[i]/(d_min-0.5))
            sigma_log += np.log(degree[i])
    
    #print(n)
    if n==0:
        return 0
    alpha=1+n/sigma_log_div
    return n * np.log(alpha) + n * alpha * np.log(d_min) - (alpha + 1) * sigma_log

def compute_log_likelihood_c(degree_a,degree_b, d_min=1):
    n=0
    sigma_log_div=0
    sigma_log=0
    for i in range(len(degree_a)):
        if degree_a[i]>=d_min:
            n+=1
            sigma_log_div+=np.log(degree_a[i]/(d_min-0.5))
            sigma_log += np.log(degree_a[i])
    for i in range(len(degree_b)):
        if degree_b[i]>=d_min:
            n+=1
            sigma_log_div+=np.log(degree_b[i]/(d_min-0.5))
            sigma_log += np.log(degree_b[i])
    #print(n)
    if n==0:
        return 0
    alpha=1+n/sigma_log_div
    
    return n * np.log(alpha) + n * alpha * np.log(d_min) - (alpha + 1) * sigma_log

def likelyhood(degree, new_d,d_min=1):
    
    return -2*compute_log_likelihood_c(degree,new_d,d_min)+2*(compute_log_likelihood(new_d,d_min)+compute_log_likelihood(degree,d_min))
