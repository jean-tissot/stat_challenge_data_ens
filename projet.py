import os,h5py,keras
os.chdir("D:\\fichiers\\Documents\\papiers\\scolarités\\bac+4\\COURS\\stat\\projet stat\\data")

x_test_file=h5py.File("X_test_new.h5",'r')
x_train_file=h5py.File("X_train_new.h5",'r')
x_test=x_test_file['features']
x_train=x_train_file['features']


###
model = keras.Sequential()
model.add(
    keras.layers.Bidirectional(
      keras.layers.LSTM(
          units=128,
          input_shape=[X_train.shape[1], X_train.shape[2]]
      )
    )
)
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])