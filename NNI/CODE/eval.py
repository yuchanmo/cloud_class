import numpy as np

import keras
from keras.datasets import fashion_mnist
from keras import backend as K
from keras.models import model_from_json


K.set_image_data_format('channels_last')
NUM_CLASSES = 10

def load_mnist_data():
    '''
    Load MNIST dataset
    '''
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = (np.expand_dims(x_train, -1).astype(np.float) / 255.)[:]
    x_test = (np.expand_dims(x_test, -1).astype(np.float) / 255.)[:]
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)[:]
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)[:]

    return x_train, y_train, x_test, y_test


def eval_model(model_name):
    x_train, y_train, x_test, y_test = load_mnist_data()

    with open('{}.json'.format(model_name), 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    optimizer = keras.optimizers.Adam()
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    ## load weights into new model
    model.load_weights("{}.h5".format(model_name))

    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Final result is: %d', acc)


if __name__ == '__main__':
    eval_model('best_model')

