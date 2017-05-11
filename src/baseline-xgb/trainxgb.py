#!/usr/bin/python2
# -*- coding: utf-8 -*-

import xgboost as xgb
import os
import math

now_path = os.getcwd() + '/'
libsvm_path = now_path + '../../dataset/DS19.libsvm'
param = {
    'max_depth': 6,
    'eta': 0.05,
    'silent': 0,
    'objective': 'multi:softmax',
    'nthread': 4,
    'num_class': 7
}
num_round = 100


def splitData(libsvm_data, test_rate=0.2):
    f = open(libsvm_data, 'r')
    ftrain = open('train.txt', 'w')
    ftest = open('test.txt', 'w')
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
    return 'train.txt', 'test.txt'

train_path, test_path = splitData(libsvm_path, test_rate=0.1)
dtrain = xgb.DMatrix(train_path)
dtest = xgb.DMatrix(test_path)
bst = xgb.train(param, dtrain, num_round,
                callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
preds_test = bst.predict(dtest)
labels = dtest.get_label()
RMSE = math.sqrt(sum((preds_test[i]-labels[i])**2 for i in range(len(preds_test))) / float(len(preds_test)))

print '\nRMSE = %f' % RMSE
print 'Correct Rate = %f' % (sum(int(preds_test[i] == labels[i]) for i in xrange(len(preds_test))) / float(len(preds_test)))
print '%d / %d' % (sum(int(preds_test[i] == labels[i]) for i in xrange(len(preds_test))), len(preds_test))
