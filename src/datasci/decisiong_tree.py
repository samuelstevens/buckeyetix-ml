import os

from sklearn import tree
from sklearn.metrics import accuracy_score

import numpy as np

data = np.load(os.getenv("HOME") + '/tmp/normalized_data.npz')

x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

c_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=10)
c_tree = c_tree.fit(x_train, y_train)


score = c_tree.score(x_test, y_test)

print(score)

tree.export_graphviz(c_tree, out_file='tree.dot')
