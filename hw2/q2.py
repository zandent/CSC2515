# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.special import logsumexp
#from scipy.misc import logsumexp

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    N_train = x_train.shape[0]
    d_train = x_train.shape[1]
    test_datum = test_datum.reshape((test_datum.shape[0],1))
    dist = l2(test_datum.T, x_train)
    nume = -dist.T/(2*tau**2)
    deno = logsumexp(nume)
    a = np.exp(nume-deno)
    A = np.identity(N_train)
    np.fill_diagonal(A,a)
    I = np.identity(d_train)
    M1 = np.matmul(np.matmul(x_train.T,A),x_train)+lam*I
    M2 = np.matmul(np.matmul(x_train.T,A),y_train)
    w = np.linalg.solve(M1,M2)
    y_pred = np.matmul(test_datum.T,w)
    return y_pred
def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    valid_batch = int(x.shape[0] * val_frac)
    np.random.seed(45689)
    rnd_idx = np.arange(x.shape[0])
    np.random.shuffle(rnd_idx)
    x_val = x[rnd_idx[:valid_batch]]
    x_train = x[rnd_idx[valid_batch:]]
    y_val = y[rnd_idx[:valid_batch]]
    y_train = y[rnd_idx[valid_batch:]]
    loss_tr = []
    loss_val = []
    for i in taus:
        print("-------------- tau is ", i," --------------")
        y_pred = []
        y_val_pred = []
        for j in range(x_train.shape[0]):
            y_pred.append(LRLS(x_train[j,:].T,x_train,y_train,i))
        for j in range(x_val.shape[0]):
            y_val_pred.append(LRLS(x_val[j,:].T,x_train,y_train,i))
        print("loss_tr is ", 0.5*np.mean((np.asarray(y_pred)-y_train.reshape((y_train.shape[0],1)))**2))
        print("loss_val is ", 0.5*np.mean((np.asarray(y_val_pred)-y_val.reshape((y_val.shape[0],1)))**2))
        loss_tr.append( 0.5*np.mean((np.asarray(y_pred)-y_train.reshape((y_train.shape[0],1)))**2))
        loss_val.append( 0.5*np.mean((np.asarray(y_val_pred)-y_val.reshape((y_val.shape[0],1)))**2))

    return loss_tr, loss_val

if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,20)
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(taus,train_losses,label="train loss")
    plt.semilogx(taus,test_losses,label="validation loss")
    plt.legend()
    plt.show()

