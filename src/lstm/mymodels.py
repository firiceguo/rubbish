#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD


"""Models from [keras](https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model)
"""


def lstm(max_features=96):
    """Best model: Sequence classification with LSTM
    Test score: 0.21813316655
    Test accuracy: 0.910412993823
    
    :param max_features: default = 96
    :return keras model: model
    """
    model = Sequential()
    model.add(Embedding(max_features, output_dim=256))
    model.add(LSTM(128))
    model.add(Dropout(0.9))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def conv1d(seq_length=96):
    """Error model: Sequence classification with 1D convolutions.
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
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


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
    model.add(Dense(64, activation='relu', input_dim=dim))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def lstm_stack(timesteps=8, data_dim=88):
    """Stacked LSTM for sequence classification
    Test score: 1.05636618459
    Test accuracy: 0.692350956063
    
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
