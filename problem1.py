import math
import numpy as np
from collections import Counter
#-------------------------------------------------------------------------
'''
    Problem 1: k nearest neighbor 
    In this problem, you will implement a classification method using k nearest neighbors.
    The main goal of this problem is to get familiar with the basic settings of classification problems. 
    KNN is a simple method for classification problems.
    You could test the correctness of your code by typing `nosetests test1.py` in the terminal.
'''

#--------------------------
def compute_distance(Xtrain, Xtest):
    '''
        compute the Euclidean distance between instances in a test set and a training set 
        Input:
            Xtrain: the feature matrix of the training dataset, a float python matrix of shape (n_train by p). Here n_train is the number of data instance in the training set, p is the number of features/dimensions.
            Xtest: the feature matrix of the test dataset, a float python matrix of shape (n_test by p). Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
        Output:
            D: the distance between instances in Xtest and Xtrain, a float python matrix of shape (ntest, ntrain), the (i,j)-th element of D represents the Euclidean distance between the i-th instance in Xtest and j-th instance in Xtrain.
        Note: youcannot use any existing function for euclidean distance, implement with only basic numpy functions, such as dot, multiply
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    train = np.array(Xtrain)
    test = np.array(Xtest)
    D = np.zeros((len(test), len(train)))
    row_sqr_sum = 0.0

    for i,train_row in enumerate(train):
        for j,test_row in enumerate(test):
            for k in range(len(train_row)):
                row_sqr_sum += ((train_row[k]-test_row[k])**2)
            row_result = math.sqrt(row_sqr_sum)
            D[j][i] = row_result
            row_sqr_sum = 0.0

    #########################################
    return D 



#--------------------------
def k_nearest_neighbor(Xtrain, Ytrain, Xtest, K = 3):
    '''
        compute the labels of test data using the K nearest neighbor classifier.
        Input:
            Xtrain: the feature matrix of the training dataset, a float numpy matrix of shape (n_train by p). Here n_train is the number of data instance in the training set, p is the number of features/dimensions.
            Ytrain: the label vector of the training dataset, an integer python list of length n_train. Each element in the list represents the label of the training instance. The values can be 0, ..., or num_class-1. num_class is the number of classes in the dataset.
            Xtest: the feature matrix of the test dataset, a float python matrix of shape (n_test by p). Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
            K: the number of neighbors to consider for classification.
        Output:
            Ytest: the predicted labels of test data, an integer numpy vector of length ntest.
        Note: you cannot use any existing package for KNN classifier.
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    train = np.array(Xtrain)
    test = np.array(Xtest)
    Y_train = np.array(Ytrain)
    row_sqr_sum = 0.0
    Ytest = np.zeros(len(test))

    for j, test_row in enumerate(test):
        distance = np.zeros(len(train))
        for i,train_row in enumerate(train):
            for b in range(len(train_row)):
                row_sqr_sum += ((train_row[b]-test_row[b])**2)
            row_result = math.sqrt(row_sqr_sum)
            row_sqr_sum = 0.0
            distance[i] = row_result
        indices = distance.argsort()[:K]
        print indices
        top_k = np.zeros(K)
        for m,val in enumerate(indices):
            top_k[m] = Y_train[val]
        print top_k
        counts = np.bincount(top_k.astype(int))
        most_common = np.argmax(counts)
        print most_common
        Ytest[j] = most_common

    print Ytest
    #########################################
    return Ytest 

