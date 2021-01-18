from read_data import dataread
from utils import vector_generator

def datatreat():
    x_test, x_train, y_train, N_test, N_train = dataread()

    # treat data here (create a vector with data)
    shape=x_test.shape
    x_test = [vector_generator(data) for data in x_test]
    x_train = [vector_generator(data) for data in x_train]
    data_shape=shape[1]*shape[2]*shape[3]

    return x_test, x_train, y_train, N_test, N_train, data_shape

#éventuellement créer autant d'autre fonction qu'il y a de méthode de traitement à tester
