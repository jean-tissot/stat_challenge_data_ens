import os,h5py,keras,pandas as pd
os.chdir("data")

x_test_file=h5py.File("X_test_new.h5",'r')
x_train_file=h5py.File("X_train_new.h5",'r')
x_test=x_test_file['features']
x_train=x_train_file['features']
N_test=x_test.shape[0]
N_train=x_train.shape[1]
y_test=pd.read_csv("y_lol_kSI9ffn.csv").label
y_train=pd.read_csv("y_train_AvCsavx.csv").label


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
model.add(keras.layers.Dense(y_train.shape[0], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

