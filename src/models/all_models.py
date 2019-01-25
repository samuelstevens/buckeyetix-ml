import multiprocessing
from multiprocessing import Pool
import sys

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adagrad, SGD, Adadelta, Adam, Adamax, Nadam


def load_data():
    data = np.load('/tmp/ml/normalized_data.npz')

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')

    return x_train, y_train, x_test, y_test


def get_model(x_train, y_train, hidden_layers=1, use_dropout=True, model_type=keras.optimizers.RMSprop, dropout=0.2):
    num_classes = y_train.shape[1]

    model = Sequential()

    for i in range(hidden_layers):
        model.add(Dense(5, activation='relu', input_shape=(x_train.shape[1],)))
        if use_dropout:
            model.add(Dropout(dropout))

    model.add(Dense(num_classes, activation='softmax') )

    model.compile(loss='categorical_crossentropy',
                  optimizer=model_type(),
                  metrics=['accuracy'])

    return model

def train_model(model, x_train, y_train, x_test, y_test, epochs):
    batch_size = x_train.shape[0]

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=(x_test, y_test))

    return

def test_model_type(model_type, x_train, y_train, x_test, y_test):
    epochs = 400

    max_accuracy = 0.0
    optimal_layers = 0
    optimal_dropout = 0.0

    for layers in range(0, 5): # number of layers
        dropout = 0.0
        for dropout_increment in range(0, 1): # reset to 5 for proper runs
            dropout += dropout_increment * 0.01

            model = get_model(
                x_train=x_train,
                y_train=y_train,
                hidden_layers=layers,
                use_dropout=True,
                model_type=model_type,
                dropout=dropout
            )

            try:
                train_model(model, x_train, y_train, x_test, y_test, epochs)


                score = model.evaluate(x_test, y_test, verbose=0)

                if score[1] > max_accuracy:
                    max_accuracy = score[1]
                    optimal_layers = layers
                    optimal_dropout = dropout
            except Exception as e:
                print(e)

        print('Trained model: ' + str(model_type) + ' with ' + str(layers) + ' hidden layer(s).')

    return max_accuracy, optimal_layers, optimal_dropout

def callback(result):
    # print(result)
    # with open('/tmp/ml/results.txt', 'a') as results_file:
    #     results_file.write(str(result))
    pass

def main():
    x_train, y_train, x_test, y_test = load_data()

    model_types = [
        keras.optimizers.RMSprop,
        keras.optimizers.Adagrad,
        keras.optimizers.SGD,
        keras.optimizers.Adadelta,
        keras.optimizers.Adam,
        keras.optimizers.Adamax,
        keras.optimizers.Nadam
    ]

    model_results = []

    max_accuracy = 0.0
    best_model = model_types[0]
    optimal_layers = 0
    optimal_dropout = 0.0

    pool = Pool()

    for model_type in model_types:
        model_results.append(pool.apply_async(test_model_type, (model_type, x_train, y_train, x_test, y_test), callback=callback))

    while 1:
        try:
            answers = []

            for result in model_results:
                answers.append(result.get(0.02))

            for (index, answer) in enumerate(answers):
                if answer[0] > max_accuracy:
                    best_model = model_types[index]
                    max_accuracy = answer[0]
                    optimal_layers = answer[1]
                    optimal_dropout = answer[2]

            break
        except multiprocessing.TimeoutError:
            pass



    print(best_model, optimal_layers, optimal_dropout)

    print('Accuracy: ' + str(max_accuracy))

main()
