import torch
from helper import load_data
from solution import PCA, AE, frobeniu_norm_error
import numpy as np
import os


def test_pca(A, p):
    pca = PCA(A, p)
    Ap, G = pca.get_reduced()
    A_re = pca.reconstruction(Ap)
    error = frobeniu_norm_error(A, A_re)
    print('PCA-Reconstruction error for {k} components is'.format(k=p), error)
    return G

def test_ae(A, p):
    model = AE(d_hidden_rep=p)
    model.train(A, A, 128, 300)
    A_re = model.reconstruction(A)
    final_w = model.get_params()
    error = frobeniu_norm_error(A, A_re)
    print('AE-Reconstruction error for {k}-dimensional hidden representation is'.format(k=p), error)
    return final_w

if __name__ == '__main__':
    dataloc = "../data/USPS.mat"
    A = load_data(dataloc)
    A = A.T
    ## Normalize A
    A = A/A.max()

    ### YOUR CODE HERE
    # Note: You are free to modify your code here for debugging and justifying your ideas for 5(f)
    #ps = [32, 64, 128]
    ps = [64]
    #err = []
    #Ierr = []
    for p in ps:
        #G = test_pca(A, p)
        W = test_ae(A, p)
        #err.append(frobeniu_norm_error(G, W))
        #Ierr.append(frobeniu_norm_error(G.T.dot(G), W.T.dot(W)))
    #print("Error between G and W:", err, "\n")
    
    #print("Error between GTG and WTW:", Ierr)
    ### END YOUR CODE 
