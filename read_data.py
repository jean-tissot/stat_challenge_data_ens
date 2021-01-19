import os,h5py,pandas as pd

os.chdir("data")

def dataread():
    x_valid_file=h5py.File("X_test_new.h5",'r')
    x_train_file=h5py.File("X_train_new.h5",'r')
    x_valid=x_valid_file['features']
    x=x_train_file['features']
    y=pd.read_csv("y_train_AvCsavx.csv").label
    return x, y, x_valid