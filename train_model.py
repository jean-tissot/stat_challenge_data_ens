from model import lstm_model, test_model
import keras


def train_lstm_model(input_shape, x_train, y_train):
    print("\tcreating lstm bidirectionnal model...")
    my_model = lstm_model(input_shape)

    print("\ttraining lstm bidirectionnal model...")
    my_model.fit(x_train, y_train, epochs=3)
    
    return my_model

def train_test_model(input_shape, x_train, y_train):
    print("\tcreating test model...")
    my_model = test_model(input_shape)

    print("\ttraining test model...")
    my_model.fit(x_train, y_train)

    return my_model

#éventuellement créer autant d'autre fonction qu'il y a d'autre modèles différents à entrainer
