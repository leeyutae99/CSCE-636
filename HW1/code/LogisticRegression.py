import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape

		# Initiate the W 
        self.W = np.zeros(n_features)
		# Iterate max_iter times
        for i in range(self.max_iter):
            gradient = np.zeros(n_features)
            for j in range(len(X)):
                #Take gradient of every element
                gradient_j = self._gradient(X[j], y[j])
                gradient += gradient_j
            #Take mean of gradients
            gradient = np.dot(1/len(X), gradient)
            #Add direction
            direction = -1 * gradient
            #Apply direction and learning_rate to W
            self.W += self.learning_rate*direction
		### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        ### Initiate the W
        self.W = np.zeros(n_features)
        #Iterate mas_iter times
        for i in range(self.max_iter):
            gradient = np.zeros(n_features)
            ### Choose random_j
            random_j = np.random.choice(len(X), batch_size, replace=False)
            for j in range(batch_size):
                ###take gradient of those j's
                gradient_j = self._gradient(X[random_j[j]], y[random_j[j]])
                gradient += gradient_j
            #Take mean of gradients
            gradient = np.dot(1/batch_size, gradient)
            #Add direction
            direction = -1 * gradient
            #Apply direction and learning_rate to W
            self.W += self.learning_rate*direction
		### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE
        n_samples, n_features = X.shape
        ### Initiate the W
        self.W = np.zeros(n_features)
        # Iterate max_iter times
        for i in range(self.max_iter):
            for j in range(n_samples):
                #Take Gradient
                gradient = self._gradient(X[j], y[j])
                direction = -1 * gradient                
                self.W += self.learning_rate * direction
        ### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        _g = -_y*_x/(1 + np.exp(_y*np.dot(self.W,_x)))
        return _g
		### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        prob = []
        prob_2 = []
        for i in range(n_samples):
            pred_prob = 1 / (1 + np.exp(-np.dot(self.W, X[i])))
            prob.append(pred_prob)
            prob_2.append(1 - pred_prob)
        preds_proba = np.vstack([prob, prob_2]).transpose()
        
        return preds_proba
		### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        preds = []
        for i in range(n_samples):
            if np.dot(self.W, X[i]) >= 0:
                y_pred = 1
            else:
                y_pred = -1
                preds.append(y_pred)
        return preds
		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE
        n_samples, n_features= X.shape
        correct_score = 0
        for i in range(n_samples):
            if y[i]*np.dot(self.W, X[i])>=0:
                correct_score += 1
        score = correct_score/n_samples * 100
        return score
		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

