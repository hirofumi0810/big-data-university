# -*- coding: utf-8 -*-


import numpy as np
import sys
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from data_making import *

SUBMIT = 'future-temparture-prediction/submission.dat'


def evaluation(X, y, cv, alpha, regression_type):
    """
    for evaluation
    """

    # interpolation
    imp = Imputer(strategy='mean', axis=0)
    imp.fit(X)
    X = imp.transform(X)

    if regression_type == 0:
        # library
        reg = Ridge(alpha=alpha)
        scores = cross_val_score(reg, X, y, cv=cv, scoring='mean_squared_error')
        return np.mean(map(lambda x: np.sqrt(-x), scores))

    elif regression_type == 1:
        # my implementation
        # feature_type == 1だとランク落ちで逆行列が計算できない
        return cross_validation(X, y, cv, alpha)


def prediction(X_train, X_test, y_train, alpha, regression_type):
    """
    for prediction
    """

    # interpolation
    imp = Imputer(strategy='mean', axis=0)
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    imp.fit(X_test)
    X_test = imp.transform(X_test)

    if regression_type == 0:
        # library
        reg_submit = Ridge(alpha=alpha)
        reg_submit.fit(X_train, y_train)
        y_test_pred = reg_submit.predict(X_test)

    elif regression_type == 1:
        # my implementation
        y_test_pred = myRedge(X_train, X_test, y_train, alpha)

    # save as text
    np.savetxt(SUBMIT, y_test_pred, fmt='%.10f')


def cross_validation(X, y, cv, alpha):
    """
    cross validation
    """

    if len(X) != len(y):
        exit()

    # cross validation
    scores = []
    block_length = len(X) / cv

    for i in xrange(cv):
        # divide into train and test data
        delete_block = [i * block_length + j for j in xrange(block_length)]
        X_train = X
        X_train = np.matrix(np.delete(X_train, delete_block, 0))
        y_train = y
        y_train = np.matrix(np.delete(y_train, delete_block, 0))
        X_test = np.matrix(X[i * block_length: (i + 1) * block_length])
        y_test = np.matrix(y[i * block_length: (i + 1) * block_length])

        # evaluation
        y_estimator = myRedge(X_train, X_test, y_train, alpha)
        y_diff = y_estimator - y_test.T
        score = np.sqrt(np.mean(map(lambda x: x * x, y_diff)))
        scores.append(score)

    return np.mean(scores)


def myRedge(X_train, X_test, y_train, alpha):
    """
    my Redge regression
    """

    W = ((np.linalg.inv((X_train.T).dot(X_train) + alpha *
                        np.identity(X_train.shape[1]))).dot(X_train.T)).dot(y_train.T)
    y_estimator = X_test.dot(W)

    return y_estimator


if __name__ == "__main__":
    argv = sys.argv

    if len(argv) != 3:
        print "Invalid arguments."
        print "Usage: python regression.py <regression_type> <feature_type>"
        print "regression_type := 0 => library, 1 => my implementation"
        print "feature_type := 0 => all features, 1 => selected feartures"
        exit()

    regression_type = int(argv[1])
    feature_type = int(argv[2])

    # regression_type
    if not regression_type in [0, 1]:
        print "Regression type is 0 or 1."
        print "0: library"
        print "1: my implementation"
        exit()

    # feature_type
    if feature_type == 0:
        # all features
        X_train, y = feature_all(0)
        X_test = feature_all(1)

    elif feature_type == 1:
        # selected features
        X_train, y = feature_select(0)
        X_test = feature_select(1)

    else:
        print "Feature type is 0 or 1."
        print "0: all features"
        print "1: selected features"
        exit()

    ####################
    # for evaluation
    ####################
    # grid search
    accuracies = []
    for i in xrange(100):
        # cross validation
        accuracy = evaluation(X_train, y, 10, 0.01 * i, regression_type)
        accuracies.append(accuracy)
    print 'alpha: %f' % (accuracies.index(min(accuracies)) * 0.01)
    print 'f-measure: %f' % min(accuracies)

    ####################
    # for submission
    ####################
    # prediction
    prediction(X_train, X_test, y, 0.22, regression_type)

    # 最終提出ver
    # day, hour, selected, locationはなし, alpha = 0.17
