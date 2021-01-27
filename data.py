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


def datatreat_1(X0, y0):
    X=[]
    y=[]
    for i in range(np.shape(X0)[0]):
        for j in range(np.shape(X0)[1]):
            X.append(X0[i,j,:,:])
            y.append(y0[i])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, shuffle=True)

    X_train=np.array(X_train)
    X_train = X_train.reshape(X_train.shape[0], 7, 500, 1)
    X_test=np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], 7, 500, 1)

    return X_train, X_test, np.array(y_train), np.array(y_test)


def datatreat_2():
    x, y, x_valid = dataread()

    # treat data here (create a vector with data)
    shape=x.shape
    x = [vector_generator(data) for data in x]
    x_valid = [vector_generator(data) for data in x_valid]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.6)

    return x_train, x_test, y_train, y_test, x_valid, shape[1]*shape[2]*shape[3]


def datatreat_3(X0, y0):
    X1, X_test, y1, y_test = train_test_split(X0, y0, train_size=0.8, shuffle=True)
    X_train=[]
    y_train=[]
    for i in range(np.shape(X1)[0]):
        for j in range(np.shape(X1)[1]):
            X_train.append(X1[i,j,:,:])
            y_train.append(y1[i])
    X_train=np.array(X_train)
    X_train = X_train.reshape(X_train.shape[0], 7, 500, 1)

    return X_train, np.array(X_test), np.array(y_train), np.array(y_test)


def datatreat_4(X0, y0, preprocess='None'):
    X1, X_test, y1, y_test = train_test_split(X0, y0, train_size=0.8, shuffle=True)
    
    a=X1[0,0,:,:]

    if preprocess=='Standardization':
        for i in range(X1.shape[0]):
            for j in range(40):
                X1[i,j,:,:]=np.transpose(StandardScaler().fit_transform(np.transpose(X1[i,j,:,:])))
        for i in range(X_test.shape[0]):
            for j in range(40):
                X_test[i,j,:,:]=np.transpose(StandardScaler().fit_transform(np.transpose(X_test[i,j,:,:])))
    if preprocess=='Normalization':
        for i in range(X1.shape[0]):
            for j in range(40):
                X1[i,j,:,:]=np.transpose(MinMaxScaler().fit_transform(np.transpose(X1[i,j,:,:])))
        for i in range(X_test.shape[0]):
            for j in range(40):
                X_test[i,j,:,:]=np.transpose(MinMaxScaler().fit_transform(np.transpose(X_test[i,j,:,:])))

    b=X1[0,0,:,:]

    X_train=[]
    y_train=[]
    for i in range(np.shape(X1)[0]):
        for j in range(np.shape(X1)[1]):
            X_train.append(X1[i,j,:,:])
            y_train.append(y1[i])

    X_train=np.array(X_train)
    y_train=np.array(y_train)

    prop_f=np.count_nonzero(y_train)/np.shape(y_train)[0]

    if prop_f<0.5:
        mask=[]
        for i in range(np.shape(y_train)[0]):
            if y_train[i]==0:
                mask.append(i)
        mask=np.array(mask)
        mask=np.resize(mask, (int(np.shape(mask)[0]-np.shape(y_train)[0]*(prop_f)),1))
        remove=int((np.shape(mask)[0]/np.shape(y_train)[0])*100)
        print(str(remove) + ' % des données d entraînement ont été retirées pour obtenir un dataset équilibré' )
        X_train=np.delete(X_train, mask, 0)
        y_train=np.delete(y_train, mask, 0)

    X_train = X_train.reshape(X_train.shape[0], 7, 500, 1)

    return X_train, np.array(X_test), y_train, np.array(y_test)


def datatreat_5(X0, y0, preprocess='None'):
    X1, X_test, y1, y_test = train_test_split(X0, y0, train_size=0.8, shuffle=True)
    
    a=X1[0,0,:,:]

    if preprocess=='Standardization':
        for i in range(X1.shape[0]):
            for j in range(40):
                X1[i,j,:,:]=np.transpose(StandardScaler().fit_transform(np.transpose(X1[i,j,:,:])))
        for i in range(X_test.shape[0]):
            for j in range(40):
                X_test[i,j,:,:]=np.transpose(StandardScaler().fit_transform(np.transpose(X_test[i,j,:,:])))
    if preprocess=='Normalization':
        for i in range(X1.shape[0]):
            for j in range(40):
                X1[i,j,:,:]=np.transpose(MinMaxScaler().fit_transform(np.transpose(X1[i,j,:,:])))
        for i in range(X_test.shape[0]):
            for j in range(40):
                X_test[i,j,:,:]=np.transpose(MinMaxScaler().fit_transform(np.transpose(X_test[i,j,:,:])))

    b=X1[0,0,:,:]

    X_train=[]
    y_train=[]
    for i in range(np.shape(X1)[0]):
        for j in range(np.shape(X1)[1]):
            X_train.append(X1[i,j,:,:])
            y_train.append(y1[i])

    X_train=np.array(X_train)
    y_train=np.array(y_train)

    X_train, y_train = shuffle(X_train, y_train)

    prop_f=np.count_nonzero(y_train)/np.shape(y_train)[0]

    if prop_f<0.5:
        mask=[]
        for i in range(np.shape(y_train)[0]):
            if y_train[i]==0:
                mask.append(i)
        mask=np.array(mask)
        mask=np.resize(mask, (int(np.shape(mask)[0]-np.shape(y_train)[0]*(prop_f)),1))
        remove=int((np.shape(mask)[0]/np.shape(y_train)[0])*100)
        print(str(remove) + ' % des données d entraînement ont été retirées pour obtenir un dataset équilibré' )
        X_train=np.delete(X_train, mask, 0)
        y_train=np.delete(y_train, mask, 0)

    X_train = X_train.reshape(X_train.shape[0], 7, 500, 1)

    return X_train, np.array(X_test), y_train, np.array(y_test)


def datatreat_6(X0, y0, preprocess='None', n_dup=1):
    X1, X_test, y1, y_test = train_test_split(X0, y0, train_size=0.8, shuffle=True)
    
    a=X1[0,0,:,:]

    if preprocess=='Standardization':
        for i in range(X1.shape[0]):
            for j in range(40):
                X1[i,j,:,:]=np.transpose(StandardScaler().fit_transform(np.transpose(X1[i,j,:,:])))
        for i in range(X_test.shape[0]):
            for j in range(40):
                X_test[i,j,:,:]=np.transpose(StandardScaler().fit_transform(np.transpose(X_test[i,j,:,:])))
    if preprocess=='Normalization':
        for i in range(X1.shape[0]):
            for j in range(40):
                X1[i,j,:,:]=np.transpose(MinMaxScaler().fit_transform(np.transpose(X1[i,j,:,:])))
        for i in range(X_test.shape[0]):
            for j in range(40):
                X_test[i,j,:,:]=np.transpose(MinMaxScaler().fit_transform(np.transpose(X_test[i,j,:,:])))

    b=X1[0,0,:,:]

    X_train=[]
    y_train=[]
    for i in range(np.shape(X1)[0]):
        for j in range(np.shape(X1)[1]):
            X_train.append(X1[i,j,:,:])
            y_train.append(y1[i])

    X_train=np.array(X_train)
    y_train=np.array(y_train)

    prop_f=np.count_nonzero(y_train)/np.shape(y_train)[0]

    if prop_f<0.5:
        new_X=[]
        new_y=[]
        for i in range(np.shape(y_train)[0]):
            if y_train[i]==0:
                new_X.append(X_train[i,:,:])
                new_y.append(y_train[i])
            else:
                for j in range(n_dup):
                    new_X.append(X_train[i,:,:])
                    new_y.append(y_train[i])
    
    X_train=np.array(new_X)
    y_train=np.array(new_y)

    prop_f=np.count_nonzero(y_train)/np.shape(y_train)[0]
    print("La proportion h/f obtenue après duplication des data f est maintenant de : " + str(prop_f))

    X_train, y_train = shuffle(X_train, y_train)

    X_train = X_train.reshape(X_train.shape[0], 7, 500, 1)

    return X_train, np.array(X_test), y_train, np.array(y_test)