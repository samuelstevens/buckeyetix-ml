import os

from sklearn.ensemble import BaggingClassifier

import numpy as np

data = np.load(os.getenv("HOME") + '/tmp/data.npz')

x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

train_size = x_train.shape[0]

best_accuracy = 0;
best_estimators = 50
best_samples = 50

counter = 0

for estimators in range(10, 400, 10): 
    for samples in range(100, train_size, 100):
        bgc = BaggingClassifier(
            n_estimators=estimators,
            max_samples=samples,
            n_jobs=-1
        )

        bgc.fit(x_train, y_train)

        counter += 1

        accuracy = bgc.score(x_test, y_test)

        if counter % 10 == 0:
            print(str(counter) + ' completed.')

        if accuracy > best_accuracy:
            print('New best accuracy: ' + str(accuracy))
            best_accuracy = accuracy
            best_samples = samples
            best_estimators = estimators

print(best_accuracy, best_estimators, best_samples)
# 0.8387909319899244 250 150
