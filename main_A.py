from data import dataread, datatreat_A1
from models import cnn_1, cnn_2, cnn_3
from test import test_1, test_2
from tools import save_model, save_results, plot_loss_acc_history
import numpy as np
import tensorflow as tf
from tensorflow import keras


id='cnn_1_None_remove'
epochs=25
batch_size=70
validation_split=0.1

X, y, X_final = dataread()

X_train, X_test, y_train, y_test, prop_HF = datatreat_A1(X, y, train_size=0.8, Shuffle=True, preprocess="None", ratio="50/50", balancing_method="SMOTEENN")

model = cnn_1()

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

save_model(model, id)

accuracy, roc, f1_macro, f1_wei = test_1(model, X_test, y_test, id)

save_results(id, 'datatreat_A1', 'cnn_1', epochs, batch_size, accuracy, roc, f1_macro, f1_wei, validation_split)

plot_loss_acc_history(history, id, validation_split)