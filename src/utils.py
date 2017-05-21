#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
import sys

now_path = os.getcwd() + '/'
data_path = now_path + '../dataset/delDirty.libsvm'


def splitData(file_path, test_rate=0.1, train_name='train.txt', test_name='test.txt'):
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


def loadLibsvm(path='', test_rate=0.05, val_rate=0.05):
    """Load data from libsvm file to numpy array for training, validation, test part.
    The data must be clean (have the same number of columns).
    """
    assert path, 'Please set the path.'
    test_rate = round(test_rate, 2)
    val_rate = round(val_rate, 2)
    x = []
    y = []
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            items = line.split(' ')
            y.append(float(items[0]))
            try:
                xx = map(lambda k: float(k.split(':')[1]), items[1:97])
            except:
                pass
            x.append(xx)
            line = f.readline()
    temp_x_train = []
    temp_y_train = []
    temp_x_val = []
    temp_y_val = []
    temp_x_test = []
    temp_y_test = []
    n_train = int(100 * (1 - test_rate - val_rate))
    n_test = int(100 * test_rate)
    for i in range(len(x)):
        if i % 100 < n_train:
            temp_x_train.append(x[i])
            temp_y_train.append(y[i])
        elif i % 100 < n_test + n_train:
            temp_x_test.append(x[i])
            temp_y_test.append(y[i])
        elif i % 100 < 100:
            temp_x_val.append(x[i])
            temp_y_val.append(y[i])
    x_train = np.asarray(temp_x_train, dtype=float)
    y_train = np.asarray(temp_y_train, dtype=int)
    x_val = np.asarray(temp_x_val, dtype=float)
    y_val = np.asarray(temp_y_val, dtype=int)
    x_test = np.asarray(temp_x_test, dtype=float)
    y_test = np.asarray(temp_y_test, dtype=int)
    if test_rate - 0 < 10e-4 and val_rate - 0 < 10e-4:
        return x_train, y_train
    else:
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


def printHelp():
    print("""
Usage:
python train.py [test_data_path] [Option] <Settings>
    -h  --help          Get this help
    -b  --batch_size    batch size for training, must be int, default=64
    -t  --timesteps     timesteps for LSTM model, must be int in [1, 95], default=32
    -r  --test_rate     test rate for training, must be float in (0, 1), default=0.05
    -v  --val_rate      validation rate for training, must be float in (0, 1), default=0.02
    """)


def myGetOpt(argv, batch_size=64, timesteps=32, test_rate=0.05, val_rate=0.05):
    batch_size, timesteps, test_rate, val_rate = batch_size, timesteps, test_rate, val_rate
    for i in range(len(argv)-1):
        opt = argv[i]
        arg = argv[i+1]
        if opt[0] == '-':
            if opt in ('-h', '--help'):
                printHelp()
                sys.exit(2)
            elif opt in ("-b", "--batch_size"):
                try:
                    batch_size = int(arg)
                except ValueError:
                    print('batch_size must be an int, using default: 64')
                    batch_size = 64
            elif opt in ("-t", "--timesteps"):
                try:
                    timesteps = int(arg)
                except ValueError:
                    print('timesteps must be an int, using default: 32')
                    batch_size = 32
            elif opt in ("-r", "--test_rate"):
                test_rate = float(arg)
                if test_rate > 1 or test_rate < 0:
                    print('Test rate must be in (0, 1), using default: 0.05')
                    test_rate = 0.05
            elif opt in ("-v", "--val_rate"):
                val_rate = float(arg)
                if val_rate > 1 or val_rate < 0:
                    print('Validation rate must be in (0, 1), using default: 0.05')
                    test_rate = 0.05
                if test_rate + val_rate >= 0.5:
                    print('Sum of test_rate and val_rate is too large (> 0.5),'
                          ' using default (test_rate=0.05, val_rate=0.05)')
                    test_rate = 0.05
                    val_rate = 0.05
        else:
            continue
    return batch_size, timesteps, test_rate, val_rate


if __name__ == '__main__':
    print(myGetOpt(sys.argv[1:]))
