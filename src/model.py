# TODO: run a bunch of different paramenters
# params to change:
# number of layers
# nodes per layer
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adagrad, SGD, Adadelta, Adam, Adamax, Nadam

data = np.load('/tmp/ml/normalized_data.npz')

x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

# TODO: research these values and their significance.
batch_size = x_train.shape[0]
epochs = 400

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print(x_train[0], 'example data point')

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

num_classes = y_train.shape[1]

model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(x_train.shape[1],)))
model.add( Dropout( 0.2 ) )
model.add(Dense(num_classes, activation='softmax') )

# model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adagrad(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
