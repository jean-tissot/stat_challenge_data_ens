from data import dataread, datatreat_1, datatreat_2, datatreat_3, datatreat_4, datatreat_5, datatreat_6
from models import cnn_1, cnn_2, cnn_3
from test import test_1, test_2
from tools import save_model, save_results, plot_loss_acc_history
import numpy as np
import tensorflow as tf
from tensorflow import keras


id='testing'
epochs=10
batch_size=70
validation_split=0.1

X, y, X_final = dataread()

X_train, X_test, y_train, y_test = datatreat_6(X, y, "None", 3)

model = cnn_1()

history = model.fit(X_train, np.array(y_train), epochs=epochs, batch_size=batch_size, validation_split=validation_split)

save_model(model, id)

accuracy, roc, f1_macro, f1_wei = test_1(model, X_test, y_test, id)

save_results(id, 'datatreat_6 "None" "3"', 'cnn_1', epochs, batch_size, accuracy, roc, f1_macro, f1_wei, validation_split)

plot_loss_acc_history(history, id, validation_split)