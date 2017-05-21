#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import regularizers


def mlp_softmax(dim=96):
    """Baseline: Multilayer Perceptron (MLP) for multi-class softmax classification

    :param dim: default=96
    :return keras model: model
    """
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(128, activation='relu', input_dim=dim))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='Adagrad',
                  metrics=['accuracy'])
    return model


def lstm_stack(timesteps=32, data_dim=64):
    """Stacked LSTM for sequence classification

    :param timesteps: int
    :param data_dim: int
    :return keras model: model
    """
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
                   input_shape=(timesteps, data_dim)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
    return model
