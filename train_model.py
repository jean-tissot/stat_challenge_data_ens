from treat_data import datatreat
from model import lstm_model
import keras

x_test, y_test, x_train, y_train, N_test, N_train = datatreat()

def train_lstm_model():
    my_model=lstm_model()
    # entrainer le modèle ici

#éventuellement créer autant d'autre fonction qu'il y a d'autre modèles différents à entrainer