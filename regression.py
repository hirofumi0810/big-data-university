# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

# PATH
# train
TEMPERATURE_TRAIN = 'future-temparture-prediction/Temperature_Train_Feature.tsv'
SUNDURATION_TRAIN = 'future-temparture-prediction/SunDuration_Train_Feature.tsv'
PRECIPITATION_TRAIN = 'future-temparture-prediction/Precipitation_Train_Feature.tsv'
LOCATION = 'future-temparture-prediction/Location.tsv'
TEMPERATURE_TRAIN_TARGET = 'future-temparture-prediction/Temperature_Train_Target.dat.tsv'

# test
TEMPERATURE_TEST = 'future-temparture-prediction/Temperature_Test_Feature.tsv'
SUNDURATION_TEST = 'future-temparture-prediction/SunDuration_test_Feature.tsv'
PRECIPITATION_TEST = 'future-temparture-prediction/Precipitation_test_Feature.tsv'
SUBMIT = 'future-temparture-prediction/submission.dat'


def feature_all():
    # read train data
    # temperature
    temp_train = pd.read_csv(TEMPERATURE_TRAIN, sep='\t')
    X_temp_train = temp_train.loc[
        :, ['place%d' % i for i in xrange(11)]].values

    # sunduration
    sund_train = pd.read_csv(SUNDURATION_TRAIN, sep='\t')
    X_sund_train = sund_train.loc[
        :, ['place%d' % i for i in xrange(11)]].values

    # precipitaion
    prec_train = pd.read_csv(PRECIPITATION_TRAIN, sep='\t')
    X_prec_train = prec_train.loc[
        :, ['place%d' % i for i in xrange(11)]].values

    # location
    location = pd.read_csv(LOCATION, sep='\t')
    X_location = location.loc[:, ['x', 'y', 'height']].values
    X_location = X_location.reshape(1, 33)
    X_location = np.tile(X_location, (1800, 1))

    # time
    X_time_train = temp_train.loc[:, ['year', 'day', 'hour']].values

    # targetplaceid
    X_targetplaceid = temp_train.loc[:, ['targetplaceid']].values

    # target
    y = np.loadtxt(TEMPERATURE_TRAIN_TARGET)

    # constant
    constant = np.array([1 for i in xrange(1800)])

    # construct feature matrix
    X_train = np.c_[constant, X_time_train, X_temp_train,
                    X_sund_train, X_prec_train, X_location, X_targetplaceid]
    return (X_train, y)


def feature_select():
    # read train data
    # temperature
    temp_train = pd.read_csv(TEMPERATURE_TRAIN, sep='\t')
    X_temp_train = temp_train.loc[
        :, ['place%d' % i for i in xrange(11)]].values

    # sunduration
    sund_train = pd.read_csv(SUNDURATION_TRAIN, sep='\t')
    X_sund_train = sund_train.loc[
        :, ['place%d' % i for i in xrange(11)]].values

    # precipitaion
    prec_train = pd.read_csv(PRECIPITATION_TRAIN, sep='\t')
    X_prec_train = prec_train.loc[
        :, ['place%d' % i for i in xrange(11)]].values

    # location
    location = pd.read_csv(LOCATION, sep='\t')
    X_location = location.loc[:, ['x', 'y', 'height']].values

    # time
    X_time_train = temp_train.loc[:, ['year', 'day', 'hour']].values

    # target
    y = np.loadtxt(TEMPERATURE_TRAIN_TARGET)

    # select features
    # temperature
    # X_temp_train_selected = [X_temp_train[i, i % 11] for i in range(1800)]
    # before 1 hour
    X_temp_train_selected_before = [
        X_temp_train[i, i % 11] for i in xrange(1800)]
    X_temp_train_selected_before = np.matrix(X_temp_train_selected_before).T
    # target time
    X_temp_train_selected = [(X_temp_train[i, i % 11] +
                              X_temp_train[i + 1, i % 11]) / 2 for i in xrange(1799)]
    X_temp_train_selected.append(X_temp_train[1799, 1799 % 11])
    X_temp_train_selected = np.matrix(X_temp_train_selected).T
    # after 1 hour
    X_temp_train_selected_after = [
        X_temp_train[i + 1, i % 11] for i in xrange(1799)]
    X_temp_train_selected_after.append(X_temp_train[1799, 1799 % 11])
    X_temp_train_selected_after = np.matrix(X_temp_train_selected_after).T

    # sunduration
    # X_sund_train_selected = [X_sund_train[i, i % 11] for i in range(1800)]
    # before 1 hour
    X_sund_train_selected_before = [
        X_sund_train[i, i % 11] for i in xrange(1800)]
    X_sund_train_selected_before = np.matrix(X_sund_train_selected_before).T
    # target time
    X_sund_train_selected = [(X_sund_train[i, i % 11] +
                              X_sund_train[i + 1, i % 11]) / 2 for i in xrange(1799)]
    X_sund_train_selected.append(X_sund_train[1799, 1799 % 11])
    X_sund_train_selected = np.matrix(X_sund_train_selected).T
    # after 1 hour
    X_sund_train_selected_after = [
        X_sund_train[i + 1, i % 11] for i in xrange(1799)]
    X_sund_train_selected_after.append(X_sund_train[1799, 1799 % 11])
    X_sund_train_selected_after = np.matrix(X_sund_train_selected_after).T

    # precipitaion
    # X_prec_train_selected = [X_prec_train[i, i % 11] for i in range(1800)]
    # before 1 hour
    X_prec_train_selected_before = [
        X_prec_train[i, i % 11] for i in xrange(1800)]
    X_prec_train_selected_before = np.matrix(X_prec_train_selected_before).T
    # target time
    X_prec_train_selected = [(X_prec_train[i, i % 11] +
                              X_prec_train[i + 1, i % 11]) / 2 for i in xrange(1799)]
    X_prec_train_selected.append(X_prec_train[1799, 1799 % 11])
    X_prec_train_selected = np.matrix(X_prec_train_selected).T
    # after 1 hour
    X_prec_train_selected_after = [
        X_prec_train[i + 1, i % 11] for i in xrange(1799)]
    X_prec_train_selected_after.append(X_prec_train[1799, 1799 % 11])
    X_prec_train_selected_after = np.matrix(X_prec_train_selected_after).T

    # location
    X_location_selected = [X_location[i % 11] for i in xrange(1800)]
    X_location_selected = np.matrix(X_location_selected)

    # construct feature matrix
    X_train = np.c_[X_time_train, X_temp_train_selected_before, X_temp_train_selected, X_temp_train_selected_after, X_sund_train_selected_before,
                    X_sund_train_selected, X_sund_train_selected_after, X_prec_train_selected_before, X_prec_train_selected, X_prec_train_selected_after, X_location_selected]
    # print X_train.shape

    return (X_train, y)


def evaluation(X, y, cv, alpha):
    # interpolation
    imp = Imputer(strategy='mean', axis=0)
    imp.fit(X)
    X = imp.transform(X)

    # Ridge regression
    # reg = Ridge(alpha=alpha)
    # scores = cross_val_score(reg, X, y, cv=cv, scoring='mean_squared_error')
    # return np.mean(map(lambda x: np.sqrt(-x), scores))

    # My regression
    return myRegression(X, y, cv, alpha)


def myRegression(X, y, cv, alpha):
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
        print y_train.shape
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
    X_train, y = feature_all()
    # X_train, y = feature_select()

    # grid search
    accuracies = []
    for i in xrange(100):
        # cross validation
        accuracy = evaluation(X_train, y, 10, 0.01 * i)
        accuracies.append(accuracy)
    print 'alpha: %f' % (accuracies.index(min(accuracies)) * 0.01)
    print min(accuracies)

    ####################
    # for submission
    ####################
    # # read test data
    # temperature
    # temp_test = pd.read_csv(TEMPERATURE_TEST, sep='\t')
    # X_temp_test = temp_test.loc[:, ['place%d' % i for i in range(11)]].values
    # X_temp_test_selected = [X_temp_test[i, i % 11] for i in range(1800)]
    # X_temp_test_selected = np.matrix(X_temp_test_selected).T
    # # sunduration
    # sund_test = pd.read_csv(SUNDURATION_TEST, sep='\t')
    # X_sund_test = sund_test.loc[:, ['place%d' % i for i in range(11)]].values
    # X_sund_test_selected = [X_sund_test[i, i % 11] for i in range(1800)]
    # X_sund_test_selected = np.matrix(X_sund_test_selected).T
    # # precipitaion
    # prec_test = pd.read_csv(PRECIPITATION_TEST, sep='\t')
    # X_prec_test = prec_test.loc[:, ['place%d' % i for i in range(11)]].values
    # X_prec_test_selected = [X_prec_test[i, i % 11] for i in range(1800)]
    # X_prec_test_selected = np.matrix(X_prec_test_selected).T
    # # time
    # X_time_test = temp_test.loc[:, ['year', 'day', 'hour']].values
    # # X_time_test = temp_test.loc[:, ['day', 'hour']].values

    # # construct feature matrix
    # X_test = np.c_[X_time_test, X_temp_test_selected, X_sund_test_selected,
    #                X_prec_test_selected, X_location_selected]
    # print X_test.shape

    # # interpolation
    # imp = Imputer(strategy='mean', axis=0)
    # imp.fit(X_train)
    # X_train = imp.transform(X_train)
    # imp.fit(X_test)
    # X_test = imp.transform(X_test)

    # # regression
    # reg_submit = Ridge(alpha=0.17)
    # reg_submit.fit(X_train, y)
    # y_test_pred = reg_submit.predict(X_test)

    # # save as test
    # np.savetxt(SUBMIT, y_test_pred, fmt='%.10f')

    # 最終提出
    # day, hour, selected, locationはなし, alpha = 0.17
