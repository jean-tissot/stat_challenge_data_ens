from tools import vector_generator, print_load
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def dataread():
    X_file=h5py.File("data/X_train_new.h5",'r') # X est ici au format "HDF5 dataset"
    X=np.array(X_file['features']) # X est ici au format "ndarray"
    
    X_final_file=h5py.File("data/X_test_new.h5",'r') # X_final est ici au format "HDF5 dataset"
    X_final=np.array(X_final_file['features']) # X_final est ici au "ndarray"

    y=pd.read_csv("data/y_train_AvCsavx.csv") # y est ici au format "pandas DataFrame"
    y=np.array(y['label']) # y est ici au format "ndarray"

    return X, y, X_final


def preprocess_A1(X_train, X_test, preprocess='None'):
    if preprocess == 'Standardization':
        for i in range(X_train.shape[0]):
            for j in range(40):
                X_train[i,j,:,:] = np.transpose(StandardScaler().fit_transform(np.transpose(X_train[i,j,:,:])))
        for i in range(X_test.shape[0]):
            for j in range(40):
                X_test[i,j,:,:] = np.transpose(StandardScaler().fit_transform(np.transpose(X_test[i,j,:,:])))
    if preprocess == 'Normalization':
        for i in range(X_train.shape[0]):
            for j in range(40):
                X_train[i,j,:,:] = np.transpose(MinMaxScaler().fit_transform(np.transpose(X_train[i,j,:,:])))
        for i in range(X_test.shape[0]):
            for j in range(40):
                X_test[i,j,:,:] = np.transpose(MinMaxScaler().fit_transform(np.transpose(X_test[i,j,:,:])))
    
    return X_train, X_test


def balancing_A1(X_train, y_train, ratio="base", balancing_method="duplicate/remove"):
    
    if ratio == "50/50":
        prop_f = np.count_nonzero(y_train) / y_train.shape[0]

        if prop_f < 0.5:
            if balancing_method == "remove":
                mask = []
                for i in range(y_train.shape[0]):
                    if y_train[i]==0:
                        mask.append(i)
                mask = np.array(mask)
                mask = np.resize(mask, (int(np.shape(mask)[0]-y_train.shape[0]*(prop_f)),1))
                X_train = np.delete(X_train, mask, 0)
                y_train = np.delete(y_train, mask, 0)
            
            if balancing_method == "duplicate":
                fem = []
                for i in range(y_train.shape[0]):
                    if y_train[i] == 1:
                        fem.append(i)
                add_X = np.take(X_train, fem, axis=0)
                add_y = np.take(y_train, fem, axis=0)
                for i in range(int((1-prop_f)/prop_f)):
                    X_train=np.concatenate((X_train,add_X), axis=0)
                    y_train=np.concatenate((y_train,add_y), axis=0)
            
            if balancing_method == "duplicate/remove":
                while prop_f < 0.5:
                    for i in range(y_train.shape[0]):
                        if y_train[i] == 1:
                            temp = X_train[i]
                            break
                    for i in range(y_train.shape[0]):
                        if y_train[i] == 0:
                            X_train[i] = temp
                            y_train[i] = 1
                            break
                    prop_f = np.count_nonzero(y_train)/np.shape(y_train)[0]
            
    prop_f = np.count_nonzero(y_train)/np.shape(y_train)[0]
    prop_HF = (1-prop_f) / prop_f

    print("La proportion H/F des données d'entraînement est de " + str(prop_HF))

    return X_train, y_train, prop_HF


def datatreat_A1(X0, y0, train_size=0.8, Shuffle=True, preprocess='None', ratio='base', balancing_method='duplicate/remove'):
    X1, X_test, y1, y_test = train_test_split(X0, y0, train_size=train_size, shuffle=Shuffle)

    X_train, X_test = preprocess_A1(X1, X_test, preprocess)

    X_train=[]
    y_train=[]
    for i in range(np.shape(X1)[0]):
        for j in range(np.shape(X1)[1]):
            X_train.append(X1[i,j,:,:])
            y_train.append(y1[i])
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_train, y_train = shuffle(X_train, y_train)

    X_train, y_train, prop_HF = balancing_A1(X_train, y_train, ratio, balancing_method)

    X_train, y_train = shuffle(X_train, y_train)

    X_train = X_train.reshape(X_train.shape[0], 7, 500, 1)

    return X_train, X_test, y_train, y_test, prop_HF