import h5py,pandas as pd, numpy as np


def dataread():
    X_file=h5py.File("data/X_train_new.h5",'r') # X est ici au format "HDF5 dataset"
    X=np.array(X_file['features']) # X est ici au format "ndarray"
    
    X_final_file=h5py.File("data/X_test_new.h5",'r') # X_final est ici au format "HDF5 dataset"
    X_final=np.array(X_final_file['features']) # X_final est ici au "ndarray"

    y=pd.read_csv("data/y_train_AvCsavx.csv") # y est ici au format "pandas DataFrame"
    y=np.array(y['label']) # y est ici au format "ndarray"

    return X, y, X_final