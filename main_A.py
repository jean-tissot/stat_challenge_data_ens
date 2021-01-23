from read_data import dataread
from treat_data import datatreat_1, datatreat_2
from models import cnn_1
import numpy as np
import tensorflow as tf
from tensorflow import keras

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

X, y, X_final = dataread()

X_train, X_test, y_train, y_test = datatreat_1(X, y)

X_train = X_train.reshape(X_train.shape[0], 7, 500, 1)
X_test = X_test.reshape(X_test.shape[0], 7, 500, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(y_train))
print(np.shape(y_test))

benchmark = cnn_1()

print(benchmark.summary())

benchmark.fit(X_train, y_train, epochs=5)

print("Evaluate on test data")
results = benchmark.evaluate(X_test, y_test)
print("test loss, test acc:", results)