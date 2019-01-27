from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
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

# Set up possible values of parameters to optimize over
p_grid = {
    'learning_rate': [0.025, 0.05, 0.075, 0.1, 0.125, 0.15],
    'n_estimators': [10, 20, 50, 75, 100, 125, 150, 200],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

estimator = GradientBoostingClassifier()

clf = GridSearchCV(estimator=estimator, param_grid=p_grid, cv=KFold(n_splits=4, shuffle=True), iid=False, n_jobs=-1)
clf.fit(x, y)

non_nested_score = clf.best_score_

nested_score = cross_val_score(clf, X=x, y=y, cv=KFold(n_splits=4, shuffle=True))

print(non_nested_score, nested_score)
