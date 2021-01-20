from read_data import dataread
from utils import vector_generator, print_load
from random import randint
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np

def data_treat_vector():
    x, y, x_valid = dataread()

    # treat data here (create a vector with data)
    shape=x.shape
    x = [vector_generator(data) for data in x]
    x_valid = [vector_generator(data) for data in x_valid]
    
    y=to_categorical(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.6)

    return x_train, x_test, y_train, y_test, x_valid, shape[1]*shape[2]*shape[3]

#éventuellement créer autant d'autre fonction qu'il y a de méthode de traitement à tester

def data_treat_alea_window():
    x, y, x_valid = dataread()

    x_alea_window=[]
    x_valid_alea_window=[]
    shape=x.shape
    for i in range(len(x)):
        print_load(i/(len(x)-1), "\tloading random windows from x...")
        x_alea_window.append(x[i][randint(0,shape[1]-1)])

    print("")
    for i in range(len(x_valid)):
        print_load(i/(len(x)-1), "\tloading random windows from x_valid...")
        x_valid_alea_window.append(x_valid[i][randint(0,shape[1]-1)])
    
    print("\n\tsplitting x and y in train and test vectors...")
    x_alea_window=np.array([[elem for elem in vector_generator(data)] for data in x_alea_window])
    x_valid=np.array([[elem for elem in vector_generator(data)] for data in x_valid_alea_window])
    y=to_categorical(y)
    x_train, x_test, y_train, y_test = train_test_split(x_alea_window, y, train_size = 0.6)

    return x_train, x_test, y_train, y_test, x_valid, (shape[2]*shape[3],)

