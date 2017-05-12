#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
import math
import sys

now_path = os.getcwd() + '/'
data_path = now_path + '../dataset/delDirty.libsvm'


def splitData(file_path, test_rate=0.2, train_name='train.txt', test_name='test.txt'):
    """Split 1 file to 2 files based on test_rate
    """
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


def loadLibsvm(path='', test_rate=0.2, val_rate=0.1):
    """Load data from libsvm file to numpy array for training, validation, test part.
    The data must be clean (have the same number of columns).
    """
    assert path, 'Please set the path.'
    test_rate = round(test_rate, 1)
    val_rate = round(val_rate, 1)
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
    temp_x_train = []
    temp_y_train = []
    temp_x_val = []
    temp_y_val = []
    temp_x_test = []
    temp_y_test = []
    n_train = int(10 * (1-test_rate-val_rate))
    n_test = int(10 * test_rate)
    n = 0
    for i in range(len(x)):
        if n < n_train:
            temp_x_train.append(x[i])
            temp_y_train.append(y[i])
            n += 1
        elif n < n_train + n_test:
            temp_x_test.append(x[i])
            temp_y_test.append(y[i])
            n += 1
        elif n < 10:
            temp_x_val.append(x[i])
            temp_y_val.append(y[i])
            n += 1
        else:
            n = 0
    x_train = np.asarray(temp_x_train, dtype=float)
    y_train = np.asarray(temp_y_train, dtype=int)
    x_val = np.asarray(temp_x_val, dtype=float)
    y_val = np.asarray(temp_y_val, dtype=int)
    x_test = np.asarray(temp_x_test, dtype=float)
    y_test = np.asarray(temp_y_test, dtype=int)
    return x_train, y_train, x_val, y_val, x_test, y_test


def addTimeStep(npdata, window_size=8):
    """Transfer data which have shape like (1000, 32) to (100, 8, 24).
    Where 8 is the window size, 24 = 32 - 8.
    """
    ori_shape = npdata.shape
    x = []
    for i in range(ori_shape[0]):
        temp = []
        for j in range(ori_shape[1]-window_size):
            try:
                temp.append(npdata[i][j:j+window_size])
            except:
                pass
        x.append(temp)
    temp = []
    for i in range(len(x)):
        temp.append(np.transpose(np.asarray(x[i], dtype=float)))
    return np.asarray(temp)


def getOpts(opts, args, batch_size=64, epoches=5, timesteps=16, model_name='lstm', test_rate=0.2, val_rate=0.1):
    """Deal with options.
    """
    model, batch_size, epoches, timesteps, test_rate, val_rate = model_name, batch_size, epoches, timesteps, test_rate, val_rate
    for opt, arg in opts:
        if opt == '-h':
            print('python train.py -m <model=\'lstm\'> -b <batch_size=64> -e <epoches=5> '
                  '-t <timesteps=16> -r <test_rate=0.1> -v <val_rate=0.1>')
            sys.exit(2)
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-b", "--batch_size"):
            batch_size = int(arg)
        elif opt in ("-e", "--epoches"):
            epoches = int(arg)
        elif opt in ("-t", "--timesteps"):
            timesteps = int(arg)
        elif opt in ("-r", "--test_rate"):
            test_rate = float(arg)
        elif opt in ("-v", "--val_rate"):
            val_rate = float(args[0])
    return model, batch_size, epoches, timesteps, test_rate, val_rate
