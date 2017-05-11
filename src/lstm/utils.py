#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import numpy as np
import math

now_path = os.getcwd() + '/'
data_path = now_path + '../dataset/delDirty.libsvm'


def splitData(file_path, test_rate=0.2, train_name='train.txt', test_name='test.txt'):
    f = open(file_path, 'r')
    ftrain = open(train_name, 'w')
    ftest = open(test_name, 'w')
    test_num = int(100 * test_rate)
    train_num = 100 - test_num
    i, j = train_num, test_num
    line = f.readline()
    while line:
        if i:
            ftrain.write(line)
            i -= 1
            line = f.readline()
        elif j:
            ftest.write(line)
            j -= 1
            line = f.readline()
        else:
            i, j = train_num, test_num
    ftrain.close()
    ftest.close()


def loadLibsvm(path='', test_rate=0.2):
    assert path, 'Please set the path.'
    x = []
    y = []
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            items = line.split(' ')
            y.append(float(items[0]))
            xx = map(lambda k: float(k.split(':')[1]), items[1:])
            x.append(xx)
            line = f.readline()
    train_num = int(math.ceil(len(x) * (1 - test_rate)))
    x_train = np.asarray(x[:train_num], dtype=float)
    x_test = np.asarray(x[train_num:], dtype=float)
    y_train = np.asarray(y[:train_num], dtype=float)
    y_test = np.asarray(y[train_num:], dtype=float)
    return x_train, y_train, x_test, y_test


def addTimeStep(npdata, window_size=8):
    ori_shape = npdata.shape
    x = []
    for i in xrange(ori_shape[0]):
        temp = []
        for j in xrange(ori_shape[1]-window_size):
            try:
                temp.append(npdata[i][j:j+window_size])
            except:
                pass
        x.append(temp)
    temp = []
    for i in xrange(len(x)):
        temp.append(np.transpose(np.asarray(x[i], dtype=float)))
    return np.asarray(temp)
