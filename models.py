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


def lstm_model(input_shape):
  model = keras.Sequential(
    [
      layers.LSTM(units=32, activation='relu', input_shape=input_shape),
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


def cnn_1():
  model = keras.Sequential(
    [
      layers.Conv2D(100, (3,3), padding='same', activation='relu', input_shape=(7,500,1)),
      layers.MaxPool2D((2,2), padding='same'),
      layers.Conv2D(100, (3,3), padding='same', activation='relu'),
      layers.MaxPool2D((2,2), padding='same'),
      layers.Conv2D(300, (2,3), padding='same', activation='relu'),
      layers.MaxPool2D((2,2), padding='same'),
      layers.Conv2D(400, (1,7), padding='same', activation='relu'),
      layers.MaxPool2D((1,2), padding='same'),
      layers.Conv2D(100, (1,3), padding='same', activation='relu'),
      layers.Conv2D(100, (1,3), padding='same', activation='relu'),
      layers.Flatten(),
      layers.Dense(500, activation='relu'),
      layers.Dense(1, activation='sigmoid')
    ]
  )

  model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
    metrics=['accuracy']
  )

  print(model.summary())

  return model


def cnn_2():
  model = keras.Sequential(
    [
      layers.Conv2D(100, (3,3), padding='same', activation='relu', input_shape=(7,500,1)),
      layers.MaxPool2D((2,2), padding='same'),
      layers.Conv2D(100, (3,3), padding='same', activation='relu'),
      layers.MaxPool2D((2,2), padding='same'),
      layers.Conv2D(300, (3,3), padding='same', activation='relu'),
      layers.MaxPool2D((2,2), padding='same'),
      layers.Conv2D(300, (3,3), padding='same', activation='relu'),
      layers.MaxPool2D((2,2), padding='same'),
      layers.Flatten(),
      layers.Dense(150, activation='relu'),
      layers.Dense(1, activation='sigmoid')
    ]
  )

  model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
  )

  print(model.summary())

  return model


def cnn_3():
  model = keras.Sequential(
    [
      layers.Conv2D(96, (3,3), padding='same', activation='relu', input_shape=(7,500,1)),
      layers.Conv2D(96, (3,3), padding='same', activation='relu'),
      layers.Conv2D(96, (3,3), padding='same', strides=2, activation='relu'),
      layers.Conv2D(192, (3,3), padding='same', activation='relu'),
      layers.Conv2D(192, (3,3), padding='same', activation='relu'),
      layers.Conv2D(192, (3,3), padding='same', strides=2, activation='relu'),
      layers.Flatten(),
      layers.Dense(200, activation='relu'),
      layers.Dense(1, activation='sigmoid')
    ]
  )

  model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
  )

  print(model.summary())

  return model