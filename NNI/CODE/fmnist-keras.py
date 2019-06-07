# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import logging

import os
import keras
import numpy as np
from keras import backend as K
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential

import nni

logger = logging.getLogger('fashion_mnist_keras')
K.set_image_data_format('channels_last')

H, W = 28, 28
NUM_CLASSES = 10

def create_mnist_model(hp, input_shape=(H, W, 1), num_classes=NUM_CLASSES):
    '''
    Create simple convolutional model
    '''
    conv_size = hp['conv_size']
    pool_size = hp['pool_size']
    layers = [
        Conv2D(hp['channel_1_num'], kernel_size=(conv_size, conv_size), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
        Conv2D(hp['channel_2_num'], (conv_size, conv_size), activation='relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
        Flatten(),
        Dense(hp['hidden_size'], activation='relu'),
        Dropout(rate=hp['dropout_rate']),
        Dense(num_classes, activation='softmax')
    ]

    model = Sequential(layers)

    optimizer = keras.optimizers.SGD(lr=hp['learning_rate'], momentum=0.9)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model

def load_mnist_data():
    '''
    Load MNIST dataset
    '''
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = (np.expand_dims(x_train, -1).astype(np.float) / 255.)[:]
    x_test = (np.expand_dims(x_test, -1).astype(np.float) / 255.)[:]
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)[:]
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)[:]

    logger.debug('x_train shape: %s', (x_train.shape,))
    logger.debug('x_test shape: %s', (x_test.shape,))

    return x_train, y_train, x_test, y_test

class SendMetrics(keras.callbacks.Callback):
    '''
    Keras callback to send metrics to NNI framework
    '''
    def on_epoch_end(self, epoch, logs={}):
        '''
        Run on end of each epoch
        '''
        logger.debug(logs)
        nni.report_intermediate_result(logs["val_acc"])

def train(params):
    '''
    Train model
    '''
    x_train, y_train, x_test, y_test = load_mnist_data()
    model = create_mnist_model(params)

    epochs = 10
    model.fit(x_train, y_train, batch_size=params['batch_size'], epochs=epochs, verbose=1,
        validation_data=(x_test, y_test), callbacks=[SendMetrics()])

    _, acc = model.evaluate(x_test, y_test, verbose=0)
    logger.debug('Final result is: %d', acc)
    nni.report_final_result(acc)

    model_id = nni.get_sequence_id()
    model_json = model.to_json()
    with open('./ckpt/model-{}.json'.format(model_id), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('./ckpt/model-{}.h5'.format(model_id))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--channel_1_num", type=int, default=32)
    parser.add_argument("--channel_2_num", type=int, default=64)
    parser.add_argument("--conv_size", type=int, default=5)
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)

    args, _ = parser.parse_known_args()

    try:
        # get parameters from tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(args)
        params.update(tuner_params)
        # train
        train(params)
    except Exception as e:
        logger.exception(e)
        raise
