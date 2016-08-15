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


def evaluation(X, y, cv, alpha):
    """
    evaluation
    """

    # interpolation
    imp = Imputer(strategy='mean', axis=0)
    imp.fit(X)
    X = imp.transform(X)

    # Ridge regression
    # reg = Ridge(alpha=alpha)
    # scores = cross_val_score(reg, X, y, cv=cv, scoring='mean_squared_error')
    # return np.mean(map(lambda x: np.sqrt(-x), scores))

    # my Ridge regression
    return myRedgeRegression(X, y, cv, alpha)


def prediction(X_train, X_test, y_train, alpha):
    """
    prediction
    """

    # interpolation
    imp = Imputer(strategy='mean', axis=0)
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    imp.fit(X_test)
    X_test = imp.transform(X_test)

    # Ridge regression
    reg_submit = Ridge(alpha=alpha)
    reg_submit.fit(X_train, y_train)
    y_test_pred = reg_submit.predict(X_test)

    # save as text
    np.savetxt(SUBMIT, y_test_pred, fmt='%.10f')


def myRedgeRegression(X, y, cv, alpha):
    """
    my implementation of Ridge regression
    """

    if len(X) != len(y):
        sys.exit(0)

    # cross validation
    scores = []
    for i in xrange(cv):
        block_length = len(X) / cv
        delete_block = [i * block_length + j for j in xrange(block_length)]

        X_train = X
        X_train = np.delete(X_train, delete_block, 0)
        y_train = y
        y_train = np.delete(y_train, delete_block, 0)
        X_val = X[i * block_length: (i + 1) * block_length]
        y_val = y[i * block_length: (i + 1) * block_length]

        # parameter estimation
        X_train = np.matrix(X_train)
        y_train = np.matrix(y_train)
        X_val = np.matrix(X_val)
        y_val = np.matrix(y_val)

        # print np.linalg.matrix_rank((X_train.T).dot(X_train) + alpha *
        # np.identity(X_train.shape[1]))
        # if np.linalg.matrix_rank((X_train.T).dot(X_train) + alpha *
        #                          np.identity(X_train.shape[1])) != X_train.shape[1]:
        #     continue
        # print y_train.shape
        W = ((np.linalg.inv((X_train.T).dot(X_train) + alpha *
                            np.identity(X_train.shape[1]))).dot(X_train.T)).dot(y_train.T)
        y_estimator = X_val.dot(W)
        y_diff = y_estimator - y_val.T
        score = np.sqrt(np.mean(map(lambda x: x * x, y_diff)))
        scores.append(score)

    return np.mean(scores)


if __name__ == "__main__":
    ####################
    # for evaluation
    ####################
    # data making
    X_train, y = feature_all(0)
    # X_train, y = feature_select(0)

    # grid search
    accuracies = []
    for i in xrange(100):
        # cross validation
        accuracy = evaluation(X_train, y, 10, 0.01 * i)
        accuracies.append(accuracy)
    print 'alpha: %f' % (accuracies.index(min(accuracies)) * 0.01)
    print 'f-measure: %f' % min(accuracies)

    ####################
    # for submission
    ####################
    # data making
    X_test = feature_all(1)
    # X_test = feature_select(1)

    # prediction
    prediction(X_train, X_test, y, 0.17)

    # 最終提出ver
    # day, hour, selected, locationはなし, alpha = 0.17
