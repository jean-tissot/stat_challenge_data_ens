from treat_data import datatreat
from train_model import train_lstm_model

x_test, y_test, x_train, y_train, N_test, N_train, data_shape = datatreat()

model=train_lstm_model(data_shape, x_train, y_train)
