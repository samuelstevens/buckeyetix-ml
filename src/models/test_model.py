import multiprocessing, sys, os, json

from multiprocessing import Pool
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


def load_data():
    data = np.load(os.getenv("HOME") + '/tmp/normalized_data.npz')

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    return x_train, y_train, x_test, y_test


def get_model(x_train, y_train):
    num_classes = y_train.shape[1]

    model = Sequential()

    model.add(Dense(9, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax') )

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    return model

def train_model(model, x_train, y_train, x_test, y_test, epochs):
    batch_size = x_train.shape[0]

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    return

def main():
    x_train, y_train, x_test, y_test = load_data()

    model = get_model(
        x_train=x_train,
        y_train=y_train,
    )

    train_model(model, x_train, y_train, x_test, y_test, 400)

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Accuracy:' + str(score[1]))

main()
