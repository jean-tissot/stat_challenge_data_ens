from treat_data import data_treat_vector, data_treat_alea_window
from train_model import train_lstm_model
from test_model import test_model

print("loading data...")
x_train, x_test, y_train, y_test, x_valid, data_shape = data_treat_alea_window()

print("training model...")
model = train_lstm_model(data_shape, x_train, y_train)

print("testing model...")
loss, accuracy = test_model(model, x_test, y_test)

print("results (loss, accuracy):", loss, accuracy)