#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import utils
import mymodels
from keras.utils import to_categorical

now_path = os.getcwd() + '/'
data_path = now_path + '../../dataset/'
ori_path = data_path + 'delDirty.libsvm'

if __name__ == '__main__':
    batch_size = 64
    x_train, y_train, x_val, y_val, x_test, y_test = utils.loadLibsvm(ori_path)
    ori_y_test = y_test
    y_train = to_categorical(y_train, num_classes=7)
    y_val = to_categorical(y_val, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)

    # do following if use lstm_stack until end
    #
    # timesteps = 16
    # data_dim = 96 - timesteps
    # x_train = utils.addTimeStep(x_train, window_size=timesteps)
    # x_val = utils.addTimeStep(x_val, window_size=timesteps)
    # x_test = utils.addTimeStep(x_test, window_size=timesteps)
    #
    # end

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    # model = mymodels.lstm_stack(timesteps=timesteps, data_dim=data_dim)
    model = mymodels.lstm()

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=5,
              validation_data=(x_val, y_val))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    pred = model.predict_classes(x_test, batch_size=batch_size, verbose=0)

    acc_num = sum(map(lambda x, y: x - y < 10e-4, pred, ori_y_test))
    print('Correct percentage:', acc_num, '/', len(y_test), '=', float(acc_num)/len(y_test))
    print('Test score:', score)
    print('Test accuracy:', acc)
