import os,h5py,pandas as pd

os.chdir("data")

def dataread():
    x_test_file=h5py.File("X_test_new.h5",'r')
    x_train_file=h5py.File("X_train_new.h5",'r')
    x_test=x_test_file['features']
    x_train=x_train_file['features']
    N_test=x_test.shape[0]
    N_train=x_train.shape[1]
    y_train=pd.read_csv("y_train_AvCsavx.csv").label
    return x_test, x_train, y_train, N_test, N_train