from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

import os

def load_data():
    data = np.load(os.getenv("HOME") + '/tmp/data.npz')

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    return x_train, y_train, x_test, y_test

# Number of random trials
NUM_TRIALS = 30

# Load the dataset
x_train, y_train, x_test, y_test = load_data()

x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

estimator = GradientBoostingClassifier()

score = cross_val_score(estimator, X=x, y=y, cv=KFold(n_splits=4, shuffle=True))

print(score)
