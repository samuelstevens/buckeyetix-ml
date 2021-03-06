from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from sklearn.preprocessing import StandardScaler

import os

def load_data():
    data = np.load(os.getenv("HOME") + '/tmp/data.npz')

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    return x_train, y_train, x_test, y_test

# Load the dataset
x_train, y_train, x_test, y_test = load_data()

x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

scaler = StandardScaler()
scaler.fit(x)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

estimator = LinearSVC(max_iter=100000)

estimator.fit(x_train, y_train)

print('Training score:')
print(estimator.score(x_train, y_train))
print('Test score:')
print(estimator.score(x_test, y_test))

score = cross_val_score(estimator, X=x, y=y, cv=KFold(n_splits=4, shuffle=True))
print('Cross validated score:')
print(score)
