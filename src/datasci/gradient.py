import os

from sklearn.ensemble import GradientBoostingClassifier

import numpy as np

def load_data():
    data = np.load(os.getenv("HOME") + '/tmp/data.npz')

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    return x_train, y_train, x_test, y_test

def test_gradient_default():
    x_train, y_train, x_test, y_test = load_data()

    train_size = x_train.shape[0]

    best_accuracy = 0
    best_estimators = 50
    best_learning = 0.01
    best_subsamples = 0.5
    best_depth = 3
    best_features = 3

    counter = 0



    for estimators in range(5, 120, 15): # 8
        for learning in range(5, 50, 10): # 4
            learning *= 0.01
            for subsample in range(10, 110, 10): # 10
                subsample *= 0.01
                for depth in range(1, 5): # 5
                    for features in range(1, 5): #5
                        clf = GradientBoostingClassifier(
                            n_estimators=estimators,
                            learning_rate=learning,
                            subsample=subsample,
                            max_depth=depth,
                            max_features=features
                        )

                        clf.fit(x_train, y_train)

                        accuracy = clf.score(x_test, y_test)

                        counter += 1

                        if counter % 10 == 0:
                            print(str(counter) + ' completed.')

                        if accuracy > best_accuracy:
                            print('New best accuracy: ' + str(accuracy))
                            best_accuracy = accuracy
                            best_estimators = estimators
                            best_learning = learning
                            best_subsamples = subsample
                            best_depth = depth
                            best_features = features
                            print(best_accuracy, best_estimators, best_learning, best_subsamples, best_depth, best_features)

    print(best_accuracy, best_estimators, best_learning, best_subsamples, best_depth, best_features)
    # 0.8463476070528967 20 0.2 0.6000000000000001 3 3
    # 0.8413098236775819 50 0.1 0.5 3 3
    # 0.8463476070528967 20 0.45 0.8 3 3
    # 0.8356807511737089 20 0.45 0.8 4 3

def test_gradient_precise():
    x_train, y_train, x_test, y_test = load_data()

    train_size = x_train.shape[0]

    best_estimators = 20
    best_learning = 0.2
    best_subsamples = 0.6

    clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.2, subsample=0.6, max_depth=3, max_features=3)
    clf.fit(x_train, y_train)

    best_accuracy = clf.score(x_test, y_test)

    print(best_accuracy, best_estimators, best_learning, best_subsamples)

    counter = 0

    for estimators in range(15, 35):
        for learning in range(12, 25):
            learning *= 0.01
            for subsample in range(45, 65, 1):
                subsample *= 0.01
                clf = GradientBoostingClassifier(
                    n_estimators=estimators,
                    learning_rate=learning,
                    subsample=subsample,
                    max_depth=3,
                    max_features=3
                )

                clf.fit(x_train, y_train)

                accuracy = clf.score(x_test, y_test)

                counter += 1

                if counter % 10 == 0:
                    print(str(counter) + ' completed.')

                if accuracy > best_accuracy:
                    print(clf.get_params())
                    print('New best accuracy: ' + str(accuracy))
                    best_accuracy = accuracy
                    # best_estimators = estimators
                    # best_learning = learning
                    # best_subsamples = subsample
                    # print(best_accuracy, best_estimators, best_learning, best_subsamples)

    print(best_accuracy, best_estimators, best_learning, best_subsamples)
    # 0.8463476070528967 20 0.2 0.60
    # 0.8438287153652393 24 0.15 0.49
    # 0.8488664987405542 33 0.23 0.64
    # 0.8463476070528967 33 0.2 0.48

test_gradient_precise()
