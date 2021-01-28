from data import dataread, datatreat_1, datatreat_2, datatreat_3, datatreat_4, datatreat_5, datatreat_6, datatreat_7
from models import cnn_1, cnn_2, cnn_3
from test import test_1, test_2
from tools import save_model, save_results, plot_loss_acc_history
import numpy as np
import tensorflow as tf
from tensorflow import keras


id='cnn_1_none_4'
epochs=200
batch_size=70
validation_split=0.1

X, y, X_final = dataread()

X_train, X_test, y_train, y_test = datatreat_7(X, y, "None")

model = cnn_1()

history = model.fit(X_train, np.array(y_train), epochs=epochs, batch_size=batch_size, validation_split=validation_split)

save_model(model, id)

accuracy, roc, f1_macro, f1_wei = test_1(model, X_test, y_test, id)

save_results(id, 'datatreat_7 "None"', 'cnn_1', epochs, batch_size, accuracy, roc, f1_macro, f1_wei, validation_split)

plot_loss_acc_history(history, id, validation_split)