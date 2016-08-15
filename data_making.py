# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# PATH
# for training
TEMPERATURE_TRAIN = 'future-temparture-prediction/Temperature_Train_Feature.tsv'
SUNDURATION_TRAIN = 'future-temparture-prediction/SunDuration_Train_Feature.tsv'
PRECIPITATION_TRAIN = 'future-temparture-prediction/Precipitation_Train_Feature.tsv'
LOCATION = 'future-temparture-prediction/Location.tsv'
TEMPERATURE_TRAIN_TARGET = 'future-temparture-prediction/Temperature_Train_Target.dat.tsv'

# for test
TEMPERATURE_TEST = 'future-temparture-prediction/Temperature_Test_Feature.tsv'
SUNDURATION_TEST = 'future-temparture-prediction/SunDuration_test_Feature.tsv'
PRECIPITATION_TEST = 'future-temparture-prediction/Precipitation_test_Feature.tsv'


def feature_all(submit):
    """
    return the matrix using all features
    """

    # read train data
    if submit != 1:
        temp_train = pd.read_csv(TEMPERATURE_TRAIN, sep='\t')
        sund_train = pd.read_csv(SUNDURATION_TRAIN, sep='\t')
        prec_train = pd.read_csv(PRECIPITATION_TRAIN, sep='\t')
        location = pd.read_csv(LOCATION, sep='\t')
        y = np.loadtxt(TEMPERATURE_TRAIN_TARGET)
    else:
        temp_train = pd.read_csv(TEMPERATURE_TRAIN, sep='\t')
        sund_train = pd.read_csv(SUNDURATION_TRAIN, sep='\t')
        prec_train = pd.read_csv(PRECIPITATION_TRAIN, sep='\t')
        location = pd.read_csv(LOCATION, sep='\t')

    # change data structure
    # temperature
    X_temp_train = temp_train.loc[:, ['place%d' % i for i in xrange(11)]].values

    # sunduration
    X_sund_train = sund_train.loc[:, ['place%d' % i for i in xrange(11)]].values

    # precipitaion
    X_prec_train = prec_train.loc[:, ['place%d' % i for i in xrange(11)]].values

    # location
    X_location = location.loc[:, ['x', 'y', 'height']].values
    X_location = X_location.reshape(1, 33)
    X_location = np.tile(X_location, (1800, 1))

    # time
    X_time_train = temp_train.loc[:, ['year', 'day', 'hour']].values

    # targetplaceid
    X_targetplaceid = temp_train.loc[:, ['targetplaceid']].values

    # constant
    constant = np.array([1 for i in xrange(1800)])

    # construct feature matrix
    X_train = np.c_[constant, X_time_train, X_temp_train,
                    X_sund_train, X_prec_train, X_location, X_targetplaceid]
    # print X_train.shape

    if submit != 1:
        return (X_train, y)
    else:
        return X_train


def feature_select(submit):
    """
    return the matrix using selected features
    """

    # read train data
    if submit != 1:
        temp_train = pd.read_csv(TEMPERATURE_TRAIN, sep='\t')
        sund_train = pd.read_csv(SUNDURATION_TRAIN, sep='\t')
        prec_train = pd.read_csv(PRECIPITATION_TRAIN, sep='\t')
        location = pd.read_csv(LOCATION, sep='\t')
        y = np.loadtxt(TEMPERATURE_TRAIN_TARGET)
    else:
        temp_train = pd.read_csv(TEMPERATURE_TRAIN, sep='\t')
        sund_train = pd.read_csv(SUNDURATION_TRAIN, sep='\t')
        prec_train = pd.read_csv(PRECIPITATION_TRAIN, sep='\t')
        location = pd.read_csv(LOCATION, sep='\t')

    # change data structure
    # temperature
    X_temp_train = temp_train.loc[:, ['place%d' % i for i in xrange(11)]].values

    # sunduration
    X_sund_train = sund_train.loc[:, ['place%d' % i for i in xrange(11)]].values

    # precipitaion
    X_prec_train = prec_train.loc[:, ['place%d' % i for i in xrange(11)]].values

    # location
    X_location = location.loc[:, ['x', 'y', 'height']].values

    # time
    X_time_train = temp_train.loc[:, ['year', 'day', 'hour']].values

    # select features
    # temperature
    # X_temp_train_selected = [X_temp_train[i, i % 11] for i in range(1800)]
    # before 1 hour
    X_temp_train_selected_before = [X_temp_train[i, i % 11] for i in xrange(1800)]
    X_temp_train_selected_before = np.matrix(X_temp_train_selected_before).T
    # target time
    X_temp_train_selected = [
        (X_temp_train[i, i % 11] + X_temp_train[i + 1, i % 11]) / 2 for i in xrange(1799)]
    X_temp_train_selected.append(X_temp_train[1799, 1799 % 11])
    X_temp_train_selected = np.matrix(X_temp_train_selected).T
    # after 1 hour
    X_temp_train_selected_after = [X_temp_train[i + 1, i % 11] for i in xrange(1799)]
    X_temp_train_selected_after.append(X_temp_train[1799, 1799 % 11])
    X_temp_train_selected_after = np.matrix(X_temp_train_selected_after).T

    # sunduration
    # X_sund_train_selected = [X_sund_train[i, i % 11] for i in range(1800)]
    # before 1 hour
    X_sund_train_selected_before = [X_sund_train[i, i % 11] for i in xrange(1800)]
    X_sund_train_selected_before = np.matrix(X_sund_train_selected_before).T
    # target time
    X_sund_train_selected = [
        (X_sund_train[i, i % 11] + X_sund_train[i + 1, i % 11]) / 2 for i in xrange(1799)]
    X_sund_train_selected.append(X_sund_train[1799, 1799 % 11])
    X_sund_train_selected = np.matrix(X_sund_train_selected).T
    # after 1 hour
    X_sund_train_selected_after = [X_sund_train[i + 1, i % 11] for i in xrange(1799)]
    X_sund_train_selected_after.append(X_sund_train[1799, 1799 % 11])
    X_sund_train_selected_after = np.matrix(X_sund_train_selected_after).T

    # precipitaion
    # X_prec_train_selected = [X_prec_train[i, i % 11] for i in range(1800)]
    # before 1 hour
    X_prec_train_selected_before = [X_prec_train[i, i % 11] for i in xrange(1800)]
    X_prec_train_selected_before = np.matrix(X_prec_train_selected_before).T
    # target time
    X_prec_train_selected = [
        (X_prec_train[i, i % 11] + X_prec_train[i + 1, i % 11]) / 2 for i in xrange(1799)]
    X_prec_train_selected.append(X_prec_train[1799, 1799 % 11])
    X_prec_train_selected = np.matrix(X_prec_train_selected).T
    # after 1 hour
    X_prec_train_selected_after = [X_prec_train[i + 1, i % 11] for i in xrange(1799)]
    X_prec_train_selected_after.append(X_prec_train[1799, 1799 % 11])
    X_prec_train_selected_after = np.matrix(X_prec_train_selected_after).T

    # location
    X_location_selected = [X_location[i % 11] for i in xrange(1800)]
    X_location_selected = np.matrix(X_location_selected)

    # construct feature matrix
    X_train = np.c_[X_time_train, X_temp_train_selected_before, X_temp_train_selected, X_temp_train_selected_after, X_sund_train_selected_before,
                    X_sund_train_selected, X_sund_train_selected_after, X_prec_train_selected_before, X_prec_train_selected, X_prec_train_selected_after, X_location_selected]
    # print X_train.shape

    if submit != 1:
        return (X_train, y)
    else:
        return X_train
