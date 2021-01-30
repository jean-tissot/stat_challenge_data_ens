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
    y_train=np.array(y_train)

    if ratio == "50/50": #on suppose que nb_f < nb_h (dans le cas contraire il suffirait d'inverser nb_f et nb_h, ainsi que mask_f et mask_h)
        mask_h = np.where(y_train==0)[0] #indices de tous les hommes
        mask_f = np.where(y_train==1)[0] #indices de toutes les femmes
        nb_h = len(mask_h) #nombre d'hommes
        nb_f = len(mask_f) #nombre de femmes

        if balancing_method == "remove":
            mask_h = mask_h[:-nb_f] #liste de taille (nb_h - nb_f) d'indices d'hommes ) à supprimer
            mask_f = []
        
        elif balancing_method == "duplicate":
            mask_h = []
            mask_f = np.resize(mask_f, nb_h - nb_f) #liste de taille (nb_h - nb_f) d'indices de femmes à rajouter (potentiellement répétés)
        
        else:
            nb = (nb_h - nb_f)//2
            mask_h = mask_h[:nb]
            mask_f = np.resize(mask_f, nb)
        
        add_X = X_train[mask_f,...] #liste des femmes à rajouter
        add_y = y_train[mask_f]
        X_train = np.delete(X_train, mask_h, 0) #suppression de la liste des hommes
        y_train = np.delete(y_train, mask_h, 0)
        X_train=np.concatenate((X_train, add_X), axis=0) #ajout de la liste de femmes
        y_train=np.concatenate((y_train, add_y), axis=0)
            
    prop_f = np.count_nonzero(y_train)/len(y_train)
    prop_HF = (1-prop_f) / prop_f

    print("La proportion H/F des données d'entraînement est de " + str(prop_HF))

    return X_train, y_train, prop_HF


def datatreat_A1(X0, y0, train_size=0.8, Shuffle=True, preprocess='None', ratio='base', balancing_method='duplicate/remove'):

    x_train, x_test, y_train, y_test = train_test_split(X0, y0, train_size=train_size, shuffle=Shuffle)
    
    x_train, x_test = preprocess_A1(x_train, x_test, preprocess)

    x_train=np.concatenate(x_train, axis=0)  #sépération des 40 fenêtres indépendantes (comme si chaque fenêtre correspondait à une personne)
    y_train=np.repeat(y_train, 40)  #Multiplication par 40 de chaque personne (car séparation des fenêtres)
    x_train, y_train = shuffle(x_train, y_train)

    x_train, y_train, prop_HF = balancing_A1(x_train, y_train, ratio, balancing_method)
    x_train, y_train = shuffle(x_train, y_train)

    x_train = x_train.reshape(x_train.shape[0], 7, 500, 1)  #Ajout d'une dimension aux données

    return x_train, x_test, y_train, y_test, prop_HF


def datatreat_J1(X0, y0, train_size=0.8, Shuffle=True, preprocess='None', ratio='base', balancing_method='duplicate/remove'):
    print("\tsplitting data...")
    x_train, x_test, y_train, y_test = train_test_split(X0, y0, train_size=train_size, shuffle=Shuffle)

    print("\tpreprocessing data...")
    x_train, x_test = preprocess_A1(x_train, x_test, preprocess)

    print("\tresizing data...")
    x_train=np.concatenate(x_train, axis=0)  #sépération des 40 fenêtres indépendantes (comme si chaque fenêtre correspondait à une personne)
    y_train=np.repeat(y_train, 40)  #Multiplication par 40 de chaque personne (car séparation des fenêtres)
    x_train=x_train.transpose(0,2,1)  #Echange des 2èmes et 3èmes dimensions (dimension canal de taille 7 et dimension EEG de taille 500)
    x_test=x_test.transpose(0,1,3,2)
    x_train, y_train = shuffle(x_train, y_train)

    x_train, y_train, prop_HF = balancing_A1(x_train, y_train, ratio, balancing_method)
    x_train, y_train = shuffle(x_train, y_train)

    return x_train, x_test, y_train, y_test, prop_HF