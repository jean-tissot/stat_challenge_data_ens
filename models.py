from tools import square, log
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def model_of_test(input_shape):
  model = keras.Sequential(
    [
      layers.Dense(32, activation='relu', input_shape=input_shape),
      layers.Flatten(),
      layers.Dense(1, activation='sigmoid')
    ]
  )

  model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
  )

  return model


def lstm_model(input_shape, loss='binary_crossentropy'):
  model = keras.Sequential(
    [
      layers.LSTM(256, dropout = 0, recurrent_dropout = 0, input_shape=input_shape),
      layers.Dense(64, activation = 'relu'),
      layers.Dense(1, activation='sigmoid')
    ]
  )

  model.compile(
    loss=loss,
    optimizer=keras.optimizers.Adamax(learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
    metrics=[keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.AUC(name='AUC')]
  )
  
  return model

def lstm_model_2(input_shape, loss='binary_crossentropy'):
  model = keras.Sequential(
    [
      layers.LSTM(256, input_shape=input_shape, return_sequences=True),
      layers.LSTM(128, return_sequences=True),
      layers.Flatten(),
      layers.Dense(64),
      layers.Dense(1, activation='sigmoid')
    ]
  )

  model.compile(
    loss=loss,
    optimizer=keras.optimizers.Adamax(learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
    metrics=[keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.AUC(name='AUC')]
  )
  
  return model

# CNN article Nature
def cnn_1():
  model = keras.Sequential(
    [
      layers.Conv2D(100, (3,3), padding='same', activation='relu', input_shape=(7,500,1)),
      layers.MaxPool2D((2,2), padding='same'),
      layers.Dropout(0.25),
      layers.Conv2D(100, (3,3), padding='same', activation='relu'),
      layers.MaxPool2D((2,2), padding='same'),
      layers.Dropout(0.25),
      layers.Conv2D(300, (2,3), padding='same', activation='relu'),
      layers.MaxPool2D((2,2), padding='same'),
      layers.Dropout(0.25),
      layers.Conv2D(300, (1,7), padding='same', activation='relu'),
      layers.MaxPool2D((1,2), padding='same'),
      layers.Dropout(0.25),
      layers.Conv2D(100, (1,3), padding='same', activation='relu'),
      layers.Conv2D(100, (1,3), padding='same', activation='relu'),
      layers.Flatten(),
      layers.Dense(3500, activation='relu'),
      layers.Dense(1, activation='sigmoid')
    ]
  )
  model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
    metrics=[keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.AUC(name='AUC')]
  )
  model.summary()

  return model


# CNN deep ConvNet
def cnn_4():
  model = keras.Sequential(
    [
      layers.Conv2D(filters=25, kernel_size=(1,10), strides=1, padding='same', input_shape=(7,500,1)),
      layers.Conv2D(filters=25, kernel_size=(7,1), strides=1, padding='same', use_bias=False),
      layers.BatchNormalization(momentum=0.1, epsilon=0.00001),
      layers.Activation('elu'),
      layers.MaxPool2D(pool_size=(1,3), strides=(1,3), padding='same'),
      
      layers.Dropout(0.5),
      layers.Conv2D(filters=50, kernel_size=(1,10), strides=1, padding='same', use_bias=False),
      layers.BatchNormalization(momentum=0.1, epsilon=0.00001),
      layers.Activation('elu'),
      layers.MaxPool2D(pool_size=(1,3), strides=(1,3), padding='same'),

      layers.Dropout(0.5),
      layers.Conv2D(filters=100, kernel_size=(1,10), strides=1, padding='same', use_bias=False),
      layers.BatchNormalization(momentum=0.1, epsilon=0.00001),
      layers.Activation('elu'),
      layers.MaxPool2D(pool_size=(1,3), strides=(1,3), padding='same'),

      layers.Dropout(0.5),
      layers.Conv2D(filters=200, kernel_size=(1,10), strides=1, padding='same', use_bias=False),
      layers.BatchNormalization(momentum=0.1, epsilon=0.00001),
      layers.Activation('elu'),
      layers.MaxPool2D(pool_size=(1,3), strides=(1,3), padding='same'),

      layers.Flatten(),
      layers.Dense(1, activation='sigmoid')
    ]
  )
  model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
    metrics=[keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.AUC(name='AUC')]
  )
  model.summary()

  return model


# CNN shallow ConvNet
def cnn_5():
  model = keras.Sequential(
    [
      layers.Conv2D(filters=40, kernel_size=(1,25), strides=1, padding='same', input_shape=(7,500,1)),
      layers.Conv2D(filters=40, kernel_size=(7,1), strides=1, padding='same', use_bias=False),
      layers.BatchNormalization(momentum=0.1, epsilon=0.00001),
      layers.Activation(activation=square),
      layers.AveragePooling2D(pool_size=(1,75), strides=(1,15), padding='same'),
      layers.Activation(activation=log),

      layers.Flatten(),
      layers.Dense(1, activation='sigmoid')
    ]
  )
  model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
    metrics=[keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.AUC(name='AUC')]
  )
  model.summary()

  return model