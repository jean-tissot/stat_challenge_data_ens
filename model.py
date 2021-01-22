from keras import Sequential
from keras.layers import LSTM, Dense, Flatten

def model_of_test(input_shape):
  model = Sequential(
    [
      Dense(32, activation='relu', input_shape=input_shape),
      Dense(2, activation='softmax') # 2 classes possible en sortie (homme ou femme)
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
      Dense(2, activation='softmax') # 2 classes possible en sortie (homme ou femme)
    ]
  )

  model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
  )
  
  return model

#éventuellement créer autant d'autre fonction qu'il y a d'autres modèle à essayer
