from read_data import dataread
from utils import vector_generator, print_load
import numpy as np
from sklearn.model_selection import train_test_split


def datatreat_1(X0, y0):
    X=[]
    y=[]
    for i in range(np.shape(X0)[0]):
        for j in range(np.shape(X0)[1]):
            X.append(X0[i,j,:,:])
            y.append(y0[i])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, shuffle=True)

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def datatreat_2():
    x, y, x_valid = dataread()

    # treat data here (create a vector with data)
    shape=x.shape
    x = [vector_generator(data) for data in x]
    x_valid = [vector_generator(data) for data in x_valid]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.6)

    return x_train, x_test, y_train, y_test, x_valid, shape[1]*shape[2]*shape[3]


#éventuellement créer autant d'autre fonction qu'il y a de méthode de traitement à tester