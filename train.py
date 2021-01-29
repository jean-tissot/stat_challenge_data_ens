from models import lstm_model, model_of_test
from tensorflow import keras


def train_lstm_model(input_shape, x_train, y_train):
    print("\tcreating lstm bidirectionnal model...")
    my_model = lstm_model(input_shape)

    print("\ttraining lstm bidirectionnal model...")
    my_model.fit(x_train, y_train, epochs=3)
    
    return my_model

def train_model_of_test(input_shape, x_train, y_train, epochs = 10, batch_size = 32):
    print("\tcreating test model...")
    my_model = model_of_test(input_shape)

    print("\ttraining test model...")
    my_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    return my_model

#éventuellement créer autant d'autre fonction qu'il y a d'autre modèles différents à entrainer
