#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import os

from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D


"""Models from [keras](https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model)
"""


def lstm_basic(max_features=96):
    """1st model: Sequence classification with LSTM
    Test score: -27.7870929233
    Test accuracy: 0.350956130484
    Failed!
    
    :param max_features: default = 96
    :return keras model: model
    """
    model = Sequential()
    model.add(Embedding(max_features, output_dim=256))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def conv1d(seq_length=96):
    """2nd model: Sequence classification with 1D convolutions.
    Don't support theano-0.8: `TypeError: pool_2d() got an unexpected keyword argument 'ws'`
    
    :param seq_length: default = 96
    :return: keras model: model
    """
    model = Sequential()
    # About Conv1D layer:
    # When using this layer as the first layer in a model, provide an 'input_shape' argument.
    # e.g.
    # (10, 128) for sequences of 10 vectors of 128-dimensional vectors,
    # (None, 128) for variable-length sequences of 128-dimensional vectors.
    model.add(Conv1D(64, 3, activation='relu', input_shape=(None, seq_length)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def lstm_imdb(max_features=96):
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
