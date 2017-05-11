#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import utils
import mymodels

now_path = os.getcwd() + '/'
data_path = now_path + '../../dataset/'
ori_path = data_path + 'delDirty.libsvm'

if __name__ == '__main__':
    batch_size = 64
    timesteps = 95
    data_dim = 96 - timesteps
    x_train, y_train, x_test, y_test = utils.loadLibsvm(ori_path, test_rate=0.2)
    # x_train = utils.addTimeStep(x_train, window_size=timesteps)
    # x_test = utils.addTimeStep(x_test, window_size=timesteps)

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = mymodels.lstm_imdb()

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=2,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
