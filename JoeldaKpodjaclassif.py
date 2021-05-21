#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 08:16:48 2021

@author: joelda
"""

from shapes import *
import numpy as np
import matplotlib.pyplot as plt
from graddesc import *


def grad_n(f, epsilon = 10**(-6)):
    def grad(x, param):
        y = np.zeros(x.shape)
        for i in range(x.shape[0]):
            x_temp = np.array(x)
            x_temp[i] += epsilon
            x_temp2 = np.array(x)
            x_temp2[i] -= epsilon
            y[i] = (f(x_temp, param)-f(x_temp2, param))/(2*epsilon)
        return y
    return grad

def grad_desc_n(f, param, dim, nb_iter, step = 0.01, x_0 = None):
    if x_0 is None:
        x_0 = np.random.randn(dim)
    grad_f = grad_n(f)
    x = x_0
    for i in range(nb_iter):
        dx = grad_f(x, param)*step
        x -= dx
        print(i, f(x,param))
    return np.array(x)

def grad_desc_stoch(f, param, dim, nb_iter, mini_batch = 100, step = 0.01, x_0 = None):
    if x_0 is None:
        x_0 = np.random.randn(dim)
    grad_f = grad_n(f)
    x = x_0
    for i in range(nb_iter):
        idx = np.random.randint(param.shape[0], size=mini_batch)
        param_stoch = param[idx]
        k = 1000
        step_stoch = step*k/(i+k)
        dx = grad_f(x, param_stoch)*step
        x -= dx
        if i%10 == 0: print(i, f(x,param))
    return np.array(x)


X = x_learn
Y = y_learn

nb_class = 4
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


S1 = np.column_stack((x_valid,y_valid))

T_sol1 = grad_desc_stoch(loss, S1, (dim_input+1)*nb_class, 10000, 100, step = 0.0001)

print("nb_error(T_sol1,S1) ")

print(nb_error(T_sol1,S1))

print("nb_error(T_sol1, x_test ")

print(nb_error(T_sol1, x_test))   

print("prediction(T_sol1,x_test ")  

print(prediction(T_sol1,x_test))

T_sol = grad_desc_stoch(loss, S, (dim_input+1)*nb_class, 10000, 100, step = 0.0001)

print("nb_error(T_sol,S1) ")

print(nb_error(T_sol,S))

print("nb_error(T_sol, x_test ")


print(nb_error(T_sol, x_test))  

print("prediction(T_sol,x_test ")  
  
print(prediction(T_sol,x_test))

