#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author: 
"""

import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):
	
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
    def fit_miniBGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch GD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

		### YOUR CODE HERE
        n_samples, n_features = X.shape
        y = np.zeros([n_samples, self.k])
        for index, label in enumerate(labels):
            label = int(label)
            y[index, label] = 1
        self.W = np.zeros([self.k, n_features])
        for i in range(self.max_iter):
            for global_index in range(0, n_samples, batch_size):
                gradient_accumulate = list()
                sample_size = n_samples - global_index if (global_index + batch_size > n_samples) else batch_size
                for random_j in range(global_index, global_index + sample_size):
                    gradient_accumulate.append(self._gradient(X[random_j], y[random_j]))
                gradient = np.mean(gradient_accumulate, axis=0)
                direction = -1 * gradient
                self.W = self.W + self.learning_rate * (direction)

                # all_gradients.append(batch_gradient)

        # print("All gradients in Multi:", all_gradients)

        return self
		### END YOUR CODE
    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features, k]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        p = self.softmax(np.matmul(self.W, _x))
        derivative = p - _y

        _x = _x.reshape(-1, 1)
        derivative = derivative.reshape(-1, 1)

        _g = np.matmul(derivative , _x.transpose())
        return _g
		### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

		### YOUR CODE HERE
        return np.exp(x) / np.sum(np.exp(x))
		### END YOUR CODE
    
    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features, k].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        prob = []
        for k in range(n_samples):
            prob.append(self.softmax(np.matmul(self.W,X[k])))
        preds = np.argmax(prob, axis=1)
        return preds
		### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        preds = self.predict(X)
        correct_score = (preds == labels).astype(np.float64)
        score = np.sum(correct_score) / n_samples * 100
        return score
		### END YOUR CODE

