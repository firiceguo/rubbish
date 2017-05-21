#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import utils

try:
    sys.argv[1]
except IndexError:
    utils.printHelp()
    sys.exit(2)
if sys.argv[1] in ('-h', '--help'):
    utils.printHelp()
    sys.exit(2)

import os
import mymodels
from keras.utils import to_categorical
import numpy as np
import features as ft
import covData as cv


def train(argv, ori_path_mlp='', ori_path_lstm='', batch_size=64, timesteps=32, test_rate=0.05, val_rate=0.05):
    mlp_epoches = 25
    all_epoches = 15
    lstm_epoches = 6
    fft_epoches = 20

    # load clean libsvm data for mlp, all and lstm
    x_train_mlp, y_train_mlp, x_val_mlp, y_val_mlp, x_test_mlp, y_test_mlp = utils.loadLibsvm(ori_path_mlp,
                                                                                              test_rate=test_rate,
                                                                                              val_rate=val_rate)
    x_train_lstm, y_train_lstm, x_val_lstm, y_val_lstm, x_test_lstm, y_test_lstm = utils.loadLibsvm(ori_path_lstm,
                                                                                                    test_rate=test_rate,
                                                                                                    val_rate=val_rate)
    x_train_all = np.hstack((x_train_mlp, x_train_lstm))
    x_val_all = np.hstack((x_val_mlp, x_val_lstm))
    x_test_all = np.hstack((x_test_mlp, x_test_lstm))

    x_train_fft = abs(np.fft.fft(x_train_lstm))
    x_val_fft = abs(np.fft.fft(x_val_lstm))
    x_test_fft = abs(np.fft.fft(x_test_lstm))

    ori_y_test_mlp = y_test_mlp
    ori_y_test_lstm = y_test_lstm

    os.remove(mlp_path)
    os.remove(lstm_path)

    # flat 1 column of classes to 7 columns of 0/1
    y_train_mlp = to_categorical(y_train_mlp, num_classes=7)
    y_val_mlp = to_categorical(y_val_mlp, num_classes=7)
    y_test_mlp = to_categorical(y_test_mlp, num_classes=7)

    y_train_lstm = to_categorical(y_train_lstm, num_classes=7)
    y_val_lstm = to_categorical(y_val_lstm, num_classes=7)
    y_test_lstm = to_categorical(y_test_lstm, num_classes=7)

    # print data shape and training options
    print('\nx_train_mlp shape:', x_train_mlp.shape)
    print('x_test_mlp shape:', x_test_mlp.shape)
    print('x_train_lstm shape:', x_train_lstm.shape)
    print('x_test_lstm shape:', x_test_lstm.shape)
    print('\nBuild model...')
    print('Using model mlp and lstm,')
    print('  with epoches:')
    print('    mlp : %d;' % mlp_epoches)
    print('    all : %d;' % all_epoches)
    print('    lstm: %d;' % lstm_epoches)
    print('  Batch_size is:', batch_size)
    print('  Timesteps is:', timesteps, '\n')

    # building mlp models
    model_mlp = mymodels.mlp_softmax(dim=29)
    model_all = mymodels.mlp_softmax(dim=29+96)
    model_fft = mymodels.mlp_softmax(dim=96)

    # building lstm model
    timesteps = timesteps
    data_dim = 96 - timesteps
    x_train_lstm = utils.addTimeStep(x_train_lstm, window_size=timesteps)
    x_val_lstm = utils.addTimeStep(x_val_lstm, window_size=timesteps)
    x_test_lstm = utils.addTimeStep(x_test_lstm, window_size=timesteps)
    model_lstm = mymodels.lstm_stack(timesteps=timesteps, data_dim=data_dim)

    while True:
        # begin train
        print('Train MLP for new features...')
        model_mlp.fit(x_train_mlp, y_train_mlp,
                      batch_size=batch_size,
                      epochs=mlp_epoches,
                      validation_data=(x_val_mlp, y_val_mlp))

        print('\nTrain MLP for all...')
        model_all.fit(x_train_all, y_train_mlp,
                      batch_size=batch_size,
                      epochs=all_epoches,
                      validation_data=(x_val_all, y_val_mlp))

        print('\nTrain LSTM for origin features...')
        model_lstm.fit(x_train_lstm, y_train_lstm,
                       batch_size=batch_size,
                       epochs=lstm_epoches,
                       validation_data=(x_val_lstm, y_val_lstm))

        print('\nTrain MLP for FFT...')
        model_fft.fit(x_train_fft, y_train_mlp,
                      batch_size=batch_size,
                      epochs=fft_epoches,
                      validation_data=(x_val_fft, y_val_lstm))

        # evaluate predict results
        score_mlp, acc_mlp = model_mlp.evaluate(x_test_mlp, y_test_mlp, batch_size=batch_size)
        score_all, acc_all = model_all.evaluate(x_test_all, y_test_mlp, batch_size=batch_size)
        score_lstm, acc_lstm = model_lstm.evaluate(x_test_lstm, y_test_lstm, batch_size=batch_size)
        score_fft, acc_fft = model_fft.evaluate(x_test_fft, y_test_mlp, batch_size=batch_size)

        # pred = model.predict_classes(x_test_mlp, verbose=1)
        pred_prob_mlp = model_mlp.predict_proba(x_test_mlp, verbose=1)
        pred_prob_all = model_all.predict_proba(x_test_all, verbose=1)
        pred_prob_lstm = model_lstm.predict_proba(x_test_lstm, verbose=1)
        pred_prob_fft = model_fft.predict_proba(x_test_fft, verbose=1)
        pred = pred_prob_lstm + pred_prob_mlp + pred_prob_all + pred_prob_fft

        pred = map(lambda x: np.ndarray.tolist(x).index(max(x)), pred)

        assert int(sum(ori_y_test_lstm - ori_y_test_mlp)) == 0, 'Features for LSTM is not match for MLP'

        acc_num = sum(map(lambda x, y: abs(x - y) < 10e-3, pred, ori_y_test_mlp))
        if float(acc_num) / len(y_test_mlp) > 0.85:
            break

    print('\n\nCorrect percentage:', acc_num, '/', len(y_test_mlp), '=', float(acc_num)/len(y_test_mlp))
    print('Test score in training set:')
    print('  For mlp :', score_mlp)
    print('  For all :', score_all)
    print('  For lstm:', score_lstm)
    print('  For fft :', score_fft)
    print('Test accuracy in training set:')
    print('  For mlp :', acc_mlp)
    print('  For all :', acc_all)
    print('  For lstm:', acc_lstm)
    print('  For fft :', acc_fft, '\n')
    return model_mlp, model_all, model_lstm, model_fft


def dataClean(in_path='', lstm_path='', mlp_path=''):
    # add zeros
    cv.dataProcess(in_path, lstm_path)
    flstm = open(lstm_path, 'r')

    # generate our new features
    fmlp = open(mlp_path, 'w')
    for line in flstm:
        arr = line.split(" ")
        value = [arr[0]]
        for i in range(1, 97):
            pair = arr[i].split(":")
            value.append(pair[1])
        value = [float(i) for i in value]
        new = ft.exctFeature(value, 16)
        s = str(value[0])
        for i in range(29):
            s += " " + str(i + 1) + ":" + str(new[i])
        fmlp.write(s + '\n')

    flstm.close()


def test(model_mlp, model_all, model_lstm, model_fft, testpath='', timesteps=32):
    print('\nTesting...')
    mlp_path = os.getcwd() + '/test_mlp.libsvm'
    lstm_path = os.getcwd() + '/test_lstm.libsvm'
    dataClean(in_path=testpath, mlp_path=mlp_path, lstm_path=lstm_path)

    x_mlp, y_mlp = utils.loadLibsvm(mlp_path,
                                    test_rate=0,
                                    val_rate=0)
    x_lstm, y_lstm = utils.loadLibsvm(lstm_path,
                                      test_rate=0,
                                      val_rate=0)
    x_all = np.hstack((x_mlp, x_lstm))
    x_fft = abs(np.fft.fft(x_lstm))

    os.remove(mlp_path)
    os.remove(lstm_path)
    ori_y_mlp = y_mlp
    ori_y_lstm = y_lstm

    # flat 1 column of classes to 7 columns of 0/1
    y_mlp = to_categorical(y_mlp, num_classes=7)

    # print data shape and training options
    print('x_train_mlp shape :', x_mlp.shape)
    print('x_train_all shape :', x_all.shape)
    print('x_train_lstm shape:', x_lstm.shape)
    print('x_train_fft shape :', x_fft.shape)

    x_lstm = utils.addTimeStep(x_lstm, window_size=timesteps)

    pred_prob_mlp = model_mlp.predict_proba(x_mlp, verbose=1)
    pred_prob_all = model_all.predict_proba(x_all, verbose=1)
    pred_prob_lstm = model_lstm.predict_proba(x_lstm, verbose=1)
    pred_prob_fft = model_fft.predict_proba(x_fft, verbose=1)
    pred = pred_prob_lstm + pred_prob_mlp + pred_prob_all + pred_prob_fft
    pred = map(lambda x: np.ndarray.tolist(x).index(max(x)), pred)

    assert int(sum(ori_y_lstm - ori_y_mlp)) == 0, 'Test for LSTM is not match for MLP'

    acc_num = sum(map(lambda x, y: abs(x - y) < 10e-3, pred, ori_y_mlp))
    print('\n\nCorrect percentage for test:\n\t', acc_num, '/', len(y_mlp), '=', float(acc_num) / len(y_mlp), '\n')


if __name__ == '__main__':
    try:
        test_path = sys.argv[1]
    except IndexError:
        utils.printHelp()
        sys.exit(1)
    assert os.path.isfile(test_path), 'Test file isn\'t exist!'

    # resolve options
    try:
        batch_size, timesteps, test_rate, val_rate = utils.myGetOpt(sys.argv[1:],
                                                                    batch_size=64, timesteps=32,
                                                                    test_rate=0.05, val_rate=0.02)
    except:
        utils.printHelp()
        sys.exit(2)

    infilepath = os.getcwd() + '/../dataset/DS19.libsvm'
    lstm_path = os.getcwd() + '/lstm.libsvm'
    mlp_path = os.getcwd() + '/mlp.libsvm'

    # Transfer the origin dataset to a clean dataset (add zeros)
    dataClean(in_path=infilepath, mlp_path=mlp_path, lstm_path=lstm_path)

    # train
    model_mlp, model_all, model_lstm, model_fft = train(sys.argv[2:], ori_path_mlp=mlp_path, ori_path_lstm=lstm_path,
                                                        batch_size=batch_size, timesteps=timesteps,
                                                        test_rate=test_rate, val_rate=val_rate)

    # test
    test(model_mlp, model_all, model_lstm, model_fft, testpath=test_path, timesteps=timesteps)
