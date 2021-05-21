#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 08:16:48 2021

@author: raphaelbailly
"""

import numpy as np
import matplotlib.pyplot as plt
from graddesc import *
from sklearn import datasets

data = datasets.load_digits()

X = data['data']
Y = data['target']

nb_class = 10
dim_input = X.shape[1]

S = np.column_stack((X,Y))


def nb_error(T_vec, S):    
    Y_target = S[:, -1]
    Y_pred = prediction(T_vec, S)
    return((Y_target != Y_pred).sum(), (Y_target != Y_pred).sum()/Y_target.shape[0])


def prediction(T_vec, S):
    T = T_vec.reshape(nb_class, dim_input+1)
    Y_pred = []
    for n_in in range(len(S)):
        v = np.zeros((nb_class))
        for n_out in range(nb_class):
            v[n_out] = (T[n_out, :dim_input].dot(S[n_in, :dim_input])+ T[n_out, dim_input])
        Y_pred.append(np.argmax(v))
    return(np.array(Y_pred))
    
def softmax(v):
    return(((np.exp(v).T)/(np.exp(v).sum(axis = 1))).T)

def softmax_v(v):
    return(np.exp(v)/(np.exp(v)).sum())

def one_hot(v, nb_class):
  return np.squeeze(np.eye(nb_class)[v.reshape(-1)])


def loss(T_vec, S):
    
    T = T_vec.reshape(nb_class, dim_input+1)
    
    T2 = T[:, :X.shape[1]]
    Score = S[:, :dim_input].dot(T2.T)+T[:,dim_input]
    Score_prob=softmax(Score)
    label = S[:,-1]
    label = label.astype(int)
    Score_log_prob = np.log(Score_prob)
    score = Score_log_prob*one_hot(label, nb_class)
    return(-score.sum())



T_sol = grad_desc_n(loss, S, (dim_input+1)*nb_class, 100, step = 0.0001)


    
