import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.figure(figsize=(10, 5))
    plt.scatter(X[y ==  1, 0], X[y ==  1, 1], c='red', marker='o', label='class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue',   marker='x', label='class 2',)
    plt.title('Features Visualized')
    plt.xlim([-1, 0])
    plt.ylim([-1, 0])
    plt.xlabel('Symmetry')
    plt.ylabel('Intensity')
    plt.legend()
    plt.savefig("../images/train_features.jpg")
    plt.clf()
    ### END YOUR CODE

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].

    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.figure(figsize = (10, 5))
    plt.scatter(X[y == 1, 0],X[y == 1,1],c = 'red', marker = 'o', label = 'class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue',   marker='x', label='class 2',)
    plt.title('Result Visualized')
    plt.xlim([-1,0])
    plt.ylim([-1,0])
    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    plt.legend()
    symmetry = np.array([X[:,0].min(), X[:,0].max()])
    db = (-W[0] - W[1]*symmetry)/W[2]
    plt.plot(symmetry,db,'--k')
    plt.savefig("../images/train_result_sigmoid.jpg")
    plt.clf()
    ### END YOUR CODE

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].

    Returns:
        No return. Save the plot to 'train_result_softmax.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.figure(figsize = (10, 5))
    plt.scatter(X[y == 0, 0],X[y == 0 , 1],c = 'red', marker = 'o', label = 'class 0')
    plt.scatter(X[y == 1, 0],X[y == 1 , 1],c = 'blue', marker = 'x', label = 'class 1')
    plt.scatter(X[y == 2, 0],X[y == 2 , 1],c = 'green', marker = 's', label = 'class 2')
    plt.title('Result Visualized')
    plt.xlim([-1,0])
    plt.ylim([-1,0])
    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    plt.legend()
    A = np.linspace(X[:,0].min(), X[:,0].max())
    db1 = np.zeros(A.shape)
    db2 = np.zeros(A.shape)
    for ix,x1 in enumerate(A):
        w0, w1, w2 = (W[0], W[1], W[2])
        db1[ix] = np.max([((w1[0] - w0[0]) + (w1[1] - w0[1])*x1)/(w0[2] - w1[2]), ((w2[0] - w0[0]) + (w2[1] - w0[1])*x1)/(w0[2] - w2[2])])
        db2[ix] = np.min([((w0[0] - w1[0]) + (w0[1] - w1[1])*x1)/(w1[2] - w0[2]), ((w2[0] - w1[0]) + (w2[1] - w1[1])*x1)/(w1[2] - w2[2])])
    plt.plot(A,db1,'--k')
    plt.plot(A,db2,'--k')
    plt.savefig("../images/train_result_softmax.jpg")
    plt.clf()
    ### END YOUR CODE


def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0] 

#    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check BGD, SGD, miniBGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    #earning_rate_list = [0.01,0.05,0.1,0.5,1]
    #max_iteration = [200 * i for i in [1,2,3,4,5]]
    #current_best_score = 0
    #best_learning_rate = 0
    #best_iteration = 0
    #for lr in learning_rate_list:
    #    for iteration in max_iteration:
    #        logisticR = logistic_regression(learning_rate=lr, max_iter=iteration)
    #        logisticR.fit_SGD(train_X, train_y)
    #        score = logisticR.score(valid_X, valid_y)
    #        if score > current_best_score:
    #            current_best_score = score
    #            best_learning_rate = lr
    #            best_iteration = iteration
    
    #print("Best Learning Rate is:" , best_learning_rate, "Best Iteration is:", best_iteration, "Best Score is:", current_best_score)
    
    #mini_batch = [5,10,50,100,200]
    #best_batch = 1
    #for batch in mini_batch:
    #    logisticR = logistic_regression(learning_rate=best_learning_rate, max_iter=best_iteration)
    #    logisticR.fit_miniBGD(train_X, train_y, batch)
    #    score = logisticR.score(valid_X, valid_y)
    #    if score > current_best_score:
    #        current_best_score = score
    #        best_batch = batch
    #print("Best Batch Size is :", best_batch, "The score for it is:", current_best_score)

    ### END YOUR CODE

	# Visualize the your 'best' model after training.
    #visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())

    ### YOUR CODE HERE
    best_logisticR = logistic_regression(learning_rate = 0.01, max_iter = 1000)
    best_logisticR.fit_SGD(train_X,train_y)
    visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())
    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    test_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_X_all = prepare_X(test_data)
    test_y_all, test_idx = prepare_y(test_labels)
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y == 2)] = -1
    print("The Result for the test data is:", best_logisticR.score(test_X, test_y))
    ### END YOUR CODE


    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  miniBGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    #learning_rate_list = [0.01,0.05,0.1,0.5,1]
    #max_iteration = [200 * i for i in [1,2,3,4,5]]
    #current_best_score = 0
    #best_learning_rate = 0
    #best_iteration = 0
    #for lr in learning_rate_list:
    #    for iteration in max_iteration:
    #        logisticR = logistic_regression_multiclass(learning_rate=lr, max_iter=iteration, k = 3)
    #        logisticR.fit_miniBGD(train_X, train_y, 1)
    #        score = logisticR.score(valid_X, valid_y)
    #        if score > current_best_score:
    #            current_best_score = score
    #            best_learning_rate = lr
    #            best_iteration = iteration
    
    #print("Best Learning Rate is:" , best_learning_rate, "Best Iteration is:", best_iteration, "Best Score is:", current_best_score)
    
    #mini_batch = [5,10,50,100,200]
    #best_batch = 1
    #for batch in mini_batch:
    #    logisticR = logistic_regression_multiclass(learning_rate=best_learning_rate, max_iter=best_iteration, k = 3)
    #    logisticR.fit_miniBGD(train_X, train_y, batch)
    #    score = logisticR.score(valid_X, valid_y)
    #    if score > current_best_score:
    #        current_best_score = score
    #        best_batch = batch
    #print("Best Batch Size is :", best_batch, "The score for it is:", current_best_score)
    ### END YOUR CODE

	# Visualize the your 'best' model after training.
	# visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())


    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE

    best_logistic_multi_R = logistic_regression_multiclass(learning_rate=0.5, max_iter=1000, k=3)
    best_logistic_multi_R.fit_miniBGD(train_X, train_y, 1)
    visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())

    test_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_X_all = prepare_X(test_data)
    test_y_all, other = prepare_y(test_labels)

    print("The Result for the test data is:", best_logistic_multi_R.score(test_X_all, test_y_all))

    ### END YOUR CODE


    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y == 2)] = 0
    log_multi_classifier = logistic_regression_multiclass(learning_rate = 0.02, max_iter = 10000, k = 2)
    log_multi_classifier.fit_miniBGD(train_X, train_y, 10)
    print('Softmax Classifier')
    print('\n')
    print("Weight is", log_multi_classifier.get_params())
    print("Training Accuracy:" , log_multi_classifier.score(train_X,train_y))
    print("Validation Accuracy:" , log_multi_classifier.score(valid_X,valid_y))
    print("Test Accuracy:" , log_multi_classifier.score(test_X,test_y))
    ### END YOUR CODE






    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1   

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(valid_y==2)] = -1 
    log_classifier = logistic_regression(learning_rate = 0.02, max_iter = 10000)
    log_classifier.fit_miniBGD(train_X, train_y, 10)
    print('Sigmoid Classifier')
    print('\n')
    print('Weight is', log_classifier.get_params())
    print('Training Accuracy:', log_classifier.score(train_X,train_y))
    print('Validation Accuracy:', log_classifier.score(valid_X,valid_y))
    print('Testing Accuracy:', log_classifier.score(test_X,test_y))
    ### END YOUR CODE


    ################Compare and report the observations/prediction accuracy


    '''
    Explore the training of these two classifiers and monitor the graidents/weights for each step. 
    Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
    Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
    '''
    ### YOUR CODE HERE
    train_y_multiclass = train_y.copy()
    train_y_multiclass[np.where(train_y_multiclass==-1)] = 0
    print("First Let's set the learning rate same for both")
    lr_class = logistic_regression(0.02, 1)
    lrm_class = logistic_regression_multiclass(0.02, 1, 2)
    lr_class.fit_miniBGD(train_X, train_y, 10)
    lrm_class.fit_miniBGD(train_X, train_y_multiclass, 10)
    print("Number of training = 0")
    print("Weight of sigmoid classifier", lr_class.get_params())
    print("Weight of Softmax classifier", lrm_class.get_params()[:,0],lrm_class.get_params()[:,1])
    print("w2 - w1 = ", lrm_class.get_params()[:,1] - lrm_class.get_params()[:,0])

    print("Let's set the learining rate of sigmoid classifier twice of the softmax's")
    for i in range(1,10):
        lr_class2 = logistic_regression(2 * 0.02, i)
        lrm_class2 = logistic_regression_multiclass(0.02, i, 2)
        lr_class2.fit_miniBGD(train_X, train_y, 10)
        lrm_class2.fit_miniBGD(train_X, train_y_multiclass, 10)
        print("Number of training = " + str(i))
        print("Weight of sigmoid classifier", lr_class2.get_params())
        print("Weight of Softmax classifier", lrm_class2.get_params()[:,0],lrm_class2.get_params()[:,1])
        print("w2 - w1 = ", lrm_class2.get_params()[:,1] - lrm_class2.get_params()[:,0])
    ### END YOUR CODE

    # ------------End------------
    

if __name__ == '__main__':
	main()
    
    
