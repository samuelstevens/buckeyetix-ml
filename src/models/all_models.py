import multiprocessing, sys, os, json, time, datetime

from util import get_nodes

from multiprocessing import Pool
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adagrad, SGD, Adadelta, Adam, Adamax, Nadam

central_file = os.getenv("HOME") + '/tmp/finished_models.json'
git_file = 'data/finished_models.json'

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
        if i == 0:
            model.add(Dense(nodes_per_layer[i], activation='relu', input_shape=(x_train.shape[1],)))
        else:
            model.add(Dense(nodes_per_layer[i], activation='relu'))

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

def test_model_type(model_type, nodes_per_layer, x_train, y_train, x_test, y_test):
    epochs = 400

    max_accuracy = 0.0
    optimal_dropout = 0.0

    dropout = 0.05

    layers = len(nodes_per_layer)
    for dropout_increment in range(0, 10): # reset to 10 for proper runs
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
                optimal_dropout = dropout

                print(
                    'Accuracy: ' +
                    str(round(max_accuracy * 100, 1)) + '%, ' +
                    str(model_type) + ' ' +
                    str(layers) + ' layers, ' +
                    str(round(optimal_dropout * 100, 1)) + '% dropout, ' +
                    str(nodes_per_layer) +
                    ' as node arrangement.'
                )

        except Exception as e:
            print(e)

        dropout += 0.02

    return {
        'model': model_type,
        'layers': layers,
        'accuracy': max_accuracy,
        'dropout': optimal_dropout,
        'nodes': nodes_per_layer,
    }

def finish_model(answer):
    answer['model'] = str(answer['model'])
    with open(git_file, 'a') as append_file:
         json.dump(answer, append_file)
         append_file.write(os.linesep)

def get_best_network(best_network, answer):
    if best_network is not None and best_network['accuracy'] > answer['accuracy']:
        return best_network
    else:
        return answer

def get_finished_models():
    try:
        with open(git_file, 'r') as read_file:
            finished_models = [json.loads(line) for line in read_file]
    except ValueError as e:
        finished_models = []
    except IOError as e:
        finished_models = []

    finished_models = set((i['model'] + str(i['nodes'])) for i in finished_models)

    return finished_models

def main():
    x_train, y_train, x_test, y_test = load_data()

    finished_models = get_finished_models()

    model_types = [
        keras.optimizers.RMSprop,
        keras.optimizers.Adagrad,
        keras.optimizers.SGD,
        keras.optimizers.Adadelta,
        keras.optimizers.Adam,
        keras.optimizers.Adamax,
        keras.optimizers.Nadam
    ]

    results = []

    best_network = None

    pool = Pool()

    start = time.clock()

    models_to_test = 0

    node_arrangements = get_nodes()

    for model_type in model_types: # 7 different optimizers
        for node_arrangement in node_arrangements:
            if str(model_type) + str(node_arrangement) not in finished_models:
                results.append(pool.apply_async(test_model_type, (model_type, node_arrangement, x_train, y_train, x_test, y_test)))
                models_to_test += 1

    if models_to_test == 0:
        print('Tested all models.')
        return

    print('Testing ' + str(models_to_test) + ' models of ' + str(7 * len(node_arrangements)) + ' models total.')

    while 1:
        try:
            for result in results:
                answer = result.get(0.02)
                results.remove(result)
                finish_model(answer)
                best_network = get_best_network(best_network, answer)

            break
        except multiprocessing.TimeoutError:
            pass

    finish = time.clock()

    best_network['time'] = str(finish - start)

    with open(os.getenv("HOME") + '/tmp/result-' + str(datetime.datetime.now()) + '.json', 'w') as outfile:
        json.dump(best_network, outfile)

main()
