from read_data import dataread
from utils import vector_generator
from random import randint
from sklearn.model_selection import train_test_split

def data_treat_vector():
    x, y, x_valid = dataread()

    # treat data here (create a vector with data)
    shape=x.shape
    x = [vector_generator(data) for data in x]
    x_valid = [vector_generator(data) for data in x_valid]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.6)

    return x_train, x_test, y_train, y_test, x_valid, shape[1]*shape[2]*shape[3]

#éventuellement créer autant d'autre fonction qu'il y a de méthode de traitement à tester

def data_treat_alea_window():
    x, y, x_valid = dataread()
    
    x_alea_window=[]
    x_valid_alea_window=[]
    shape=x.shape
    for data in x:
        x_alea_window.append(data[randint(0,shape[1]-1)])
    for data in x_valid:
        x_valid_alea_window.append(data[randint(0,shape[1]-1)])
    
    x_train, x_test, y_train, y_test = train_test_split(x_alea_window, y, train_size = 0.6)

    return x_train, x_test, y_train, y_test, x_valid, shape[2:]

