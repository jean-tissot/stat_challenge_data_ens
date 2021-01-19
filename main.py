from treat_data import data_treat_vector, data_treat_alea_window
from train_model import train_lstm_model

x_test, x_train, y_train, N_test, N_train, data_shape = data_treat_alea_window()

model=train_lstm_model(data_shape, x_train, y_train)
