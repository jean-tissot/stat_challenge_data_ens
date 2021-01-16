from read_data import dataread

def datatreat():
    x_test, y_test, x_train, y_train, N_test, N_train = dataread()

    # treat data here (creat a vector with data)

    return x_test, y_test, x_train, y_train, N_test, N_train

#éventuellement créer autant d'autre fonction qu'il y a de méthode de traitement à tester