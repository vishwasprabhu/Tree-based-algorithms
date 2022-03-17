import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
import pandas as pd

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))

def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)
    """
    ### BEGIN SOLUTION
    df = pd.read_csv(filename,header=None)
    X = df.iloc[:, :-1].values
    Y = np.array(df.iloc[:, -1].replace(0, -1).values,dtype=float)
    ### END SOLUTION
    return X, Y


def adaboost(X, y, num_iter, max_depth=1):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is {-1, 1}
    """
    trees = []
    trees_weights = [] 
    N, _ = X.shape
    d = np.ones(N) / N

    ### BEGIN SOLUTION
    for i in range(num_iter):
        tree = DecisionTreeClassifier(max_depth=max_depth)
        tree.fit(X, y, sample_weight=d)
        y_hat = tree.predict(X)
        err = sum((y != y_hat)*d/sum(d))
        alpha = np.log((1-err)/err) if err != 0 else 1
        d = d * np.exp(alpha * (y != y_hat))
        trees_weights.append(alpha)
        trees.append(tree)
    ### END SOLUTION
    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y
    """
    # X input, y output
    N, _ = X.shape
    y = np.zeros(N)
    ### BEGIN SOLUTION
    y_hat = [0] * X.shape[0]
    for i, tree in enumerate(trees):
        y_hat += trees_weights[i]*(tree.predict(X))
    y = np.sign(y_hat)
    ### END SOLUTION
    return y
