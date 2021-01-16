import keras

def lstm_model(N_train):
  model = keras.Sequential()
  model.add(
      keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=128,
            input_shape=[x_train.shape[1], x_train.shape[2]]
        )
      )
  )
  model.add(keras.layers.Dropout(rate=0.5))
  model.add(keras.layers.Dense(units=128, activation='relu'))
  model.add(keras.layers.Dense(N_train, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
  return model

#éventuellement créer autant d'autre fonction qu'il y a d'autres modèle à essayer