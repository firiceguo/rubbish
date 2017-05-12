#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import utils
import mymodels
import sys
import getopt
from keras.utils import to_categorical
import covData


def main(argv, model_name='lstm', ori_path='', batch_size=64, epoches=5, timesteps=32, test_rate=0.2, val_rate=0.1):
    # resolve options
    try:
        opts, args = getopt.getopt(argv, "hm:b:e:t:r:v",
                                   ["model=", "batch_size=", "epoches=", "timesteps=", "test_rate=", "val_rate="])
    except getopt.GetoptError:
        print('python train.py [data_path] -m <model=\'lstm\'> -b <batch_size=32> -e <epoches=5> '
              '-t <timesteps=16> -r <test_rate=0.1> -v <val_rate=0.1>')
        sys.exit(2)
    model_name, batch_size, epoches, timesteps, test_rate, val_rate = utils.getOpts(opts,
        args, batch_size=batch_size, epoches=epoches, timesteps=timesteps,
        model_name=model_name, test_rate=test_rate, val_rate=val_rate)

    # assert input
    assert ori_path, 'Please set dataset path!'
    assert model_name in ['lstm', 'mlp'], \
        'Don\'t have model ' + model_name + '! Only have model with -m: \'lstm\', \'mlp\'.'

    # load clean libsvm data
    batch_size = batch_size
    x_train, y_train, x_val, y_val, x_test, y_test = utils.loadLibsvm(ori_path, test_rate=test_rate, val_rate=val_rate)
    ori_y_test = y_test

    # flat 1 column of classes to 7 columns of 0/1
    y_train = to_categorical(y_train, num_classes=7)
    y_val = to_categorical(y_val, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)

    # print data shape and training options
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('Build model...')
    print('Using model', model_name, ', with epoches', epoches, ', and batch_size is', batch_size)

    # building different models
    if model_name == 'mlp':
        model = mymodels.mlp_softmax()
    elif model_name == 'lstm':
        print('Timesteps is:', timesteps)
        timesteps = timesteps
        data_dim = 96 - timesteps
        x_train = utils.addTimeStep(x_train, window_size=timesteps)
        x_val = utils.addTimeStep(x_val, window_size=timesteps)
        x_test = utils.addTimeStep(x_test, window_size=timesteps)
        model = mymodels.lstm_stack(timesteps=timesteps, data_dim=data_dim)

    # begin train
    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epoches,
              validation_data=(x_val, y_val))

    # evaluate predict results
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    pred = model.predict_classes(x_test, verbose=1)

    acc_num = sum(map(lambda x, y: abs(x - y) < 10e-3, pred, ori_y_test))
    print('\n\nCorrect percentage:', acc_num, '/', len(y_test), '=', float(acc_num)/len(y_test))
    print('Test score:', score)
    print('Test accuracy:', acc)


if __name__ == '__main__':
    path = sys.argv[1]
    infilepath = sys.argv[1]
    outfilepath = os.getcwd() + '/../../dataset/newData.libsvm'
    covData.dataProcess(infilepath, outfilepath)
    main(sys.argv[2:], ori_path=outfilepath)
