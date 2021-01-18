from model import lstm_model
import keras


def train_lstm_model(input_shape, x_train, y_train):
    my_model=lstm_model(input_shape)
    my_model.fit(x_train, y_train, epochs=3)
    # entrainer le modèle ici
    return my_model

#éventuellement créer autant d'autre fonction qu'il y a d'autre modèles différents à entrainer