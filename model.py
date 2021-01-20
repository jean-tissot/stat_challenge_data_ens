from keras import Sequential
from keras.layers import LSTM, Dense

def test_model(input_shape):
  model = Sequential(
    [
    Dense(32, activation='relu', input_shape=input_shape),
    Dense(3, activation='softmax')
    ]
  )

  model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
  )

  return model

def lstm_model(input_shape):
  model = Sequential(
    [
    LSTM(units=32, activation='relu', input_shape=input_shape),
    Dense(3, activation='softmax')
    ]
  )

  model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
  )
  
  return model

#éventuellement créer autant d'autre fonction qu'il y a d'autres modèle à essayer
