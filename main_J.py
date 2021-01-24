from data import data_treat_vector, data_treat_alea_window
from train import train_lstm_model, train_model_of_test
from test import test_model

print("loading data...")
x_train, x_test, y_train, y_test, x_valid, data_shape = data_treat_alea_window()

print("training model...")
model = train_model_of_test(data_shape, x_train, y_train, 30)

print("testing model...")
loss, accuracy = test_model(model, x_test, y_test)

print("results (loss, accuracy):", loss, accuracy)