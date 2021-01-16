from read_data import dataread
from utils import vector_generator

def datatreat():
    x_test, y_test, x_train, y_train, N_test, N_train = dataread()

    # treat data here (create a vector with data)
    x_test = vector_generator(x_test)
    x_train = vector_generator(x_train)

    return x_test, y_test, x_train, y_train, N_test, N_train

#éventuellement créer autant d'autre fonction qu'il y a de méthode de traitement à tester
