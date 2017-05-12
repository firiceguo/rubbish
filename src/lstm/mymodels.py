#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.optimizers import SGD


"""Models from [keras](https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model)
"""


def mlp_softmax(dim=96):
    """Baseline: Multilayer Perceptron (MLP) for multi-class softmax classification
    epoches = 200
    Test accuracy: 0.65298087672

    :param dim: default=96
    :return keras model: model
    """
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(128, activation='relu', input_dim=dim))
    # model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def lstm_stack(timesteps=8, data_dim=88):
    """Stacked LSTM for sequence classification
    Correct percentage: 594 / 808 = 0.735148514851
    Test score: 0.768332143821
    Test accuracy: 0.735148515442

    :param timesteps:
    :param data_dim:
    :return:
    """
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
                   input_shape=(timesteps, data_dim)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model
