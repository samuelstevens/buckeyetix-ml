import multiprocessing, sys, os, json, time

from multiprocessing import Pool
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adagrad, SGD, Adadelta, Adam, Adamax, Nadam


def load_data():
    data = np.load(os.getenv("HOME") + '/tmp/normalized_data.npz')

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def get_model(x_train, y_train, hidden_layers=1, use_dropout=True, model_type=keras.optimizers.RMSprop, dropout=0.2, nodes_per_layer=[5]):
    num_classes = y_train.shape[1]

    model = Sequential()

    for i in range(hidden_layers):
        model.add(Dense(nodes_per_layer[i], activation='relu', input_shape=(x_train.shape[1],)))
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
    optimal_nodes_per_layer = []

    for layers in range(1, 5): # number of layers 1-4
        dropout = 0.0
        for dropout_increment in range(0, 5): # reset to 5 for proper runs
            dropout += dropout_increment * 0.01

            nodes_per_layer = [3] * layers
            # makes an array of size <layers>, with default 3 nodes in each layer

            for layer_index in range(1, layers):
                for nodes in range(3, 11):
                    nodes_per_layer[layer_index] = nodes

                    try:
                        model = get_model(
                            x_train=x_train,
                            y_train=y_train,
                            hidden_layers=layers,
                            use_dropout=True,
                            model_type=model_type,
                            dropout=dropout,
                            nodes_per_layer=nodes_per_layer
                        )

                        train_model(model, x_train, y_train, x_test, y_test, epochs)

                        score = model.evaluate(x_test, y_test, verbose=0)

                        if score[1] > max_accuracy:
                            max_accuracy = score[1]
                            optimal_layers = layers
                            optimal_dropout = dropout
                            optimal_nodes_per_layer = nodes_per_layer

                            print(
                                'Accuracy is ' + str(round(max_accuracy * 100, 1)) +
                                '% using ' + str(optimal_layers) + ' layers, ' +
                                str(round(optimal_dropout * 100, 1)) + '% dropout, ' +
                                ' and ' + str(optimal_nodes_per_layer) +
                                ' as node arrangement.'
                            )

                    except Exception as e:
                        print(e)

    return max_accuracy, optimal_layers, optimal_dropout, optimal_nodes_per_layer

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
    optimal_nodes_per_layer = []

    pool = Pool()

    start = time.clock()

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
                    optimal_nodes_per_layer = answer[3]

            break
        except multiprocessing.TimeoutError:
            pass

    finish = time.clock()

    result = {
        'model': str(best_model),
        'layers': optimal_layers,
        'dropout': optimal_dropout,
        'nodes': optimal_nodes_per_layer,
        'time': str(finish - start)
    }

    with open(os.getenv("HOME") + '/tmp/result.json', 'w') as outfile:
        json.dump(result, outfile)

main()
