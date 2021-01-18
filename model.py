import keras

def lstm_model(input_shape):
  model = keras.Sequential()
  model.add(
      keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=128,
            input_shape=input_shape
        )
      )
  )
  model.add(keras.layers.Dropout(rate=0.5))
  model.add(keras.layers.Dense(units=128, activation='relu'))
  model.add(keras.layers.Dense(units=10, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
  return model

#éventuellement créer autant d'autre fonction qu'il y a d'autres modèle à essayer