from data import dataread, datatreat_1, datatreat_2, datatreat_3, datatreat_4
from models import cnn_1, cnn_2, cnn_3
from test import test_1, test_2
from tools import save_model, save_results
import numpy as np
import tensorflow as tf
from tensorflow import keras


id='cnn_1_unbalanced'

X, y, X_final = dataread()

X_train, X_test, y_train, y_test = datatreat_3(X, y)

model = cnn_1()

model.fit(X_train, np.array(y_train), epochs=200, batch_size=70, validation_split=0.1)

save_model(model, id)

accuracy, roc, f1_macro, f1_wei = test_1(model, X_test, y_test, id)

save_results(id, 'datatreat_3', 'cnn_1', 200, 70, accuracy, roc, f1_macro, f1_wei, 0.1)