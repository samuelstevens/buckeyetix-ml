import os

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import numpy as np

data = np.load(os.getenv("HOME") + '/tmp/data.npz')

x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

bdt = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    algorithm="SAMME.R",
    n_estimators=200,
    learning_rate=0.5
)

bdt.fit(x_train, y_train)

print(bdt.score(x_test, y_test))
