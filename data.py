from tools import vector_generator, print_load
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


def dataread():
    X_file=h5py.File("data/X_train_new.h5",'r') # X est ici au format "HDF5 dataset"
    X=np.array(X_file['features']) # X est ici au format "ndarray"
    
    X_final_file=h5py.File("data/X_test_new.h5",'r') # X_final est ici au format "HDF5 dataset"
    X_final=np.array(X_final_file['features']) # X_final est ici au "ndarray"

    y=pd.read_csv("data/y_train_AvCsavx.csv") # y est ici au format "pandas DataFrame"
    y=np.array(y['label']) # y est ici au format "ndarray"

    return X, y, X_final


def preprocess_A1(X_train, X_test, preprocess=None):
    
    if preprocess:
        transform = StandardScaler().fit_transform if 'tand' in preprocess else MinMaxScaler().fit_transform # Standardization ou Scaling
        for j in range(40):
            for i in range(X_train.shape[0]):
                X_train[i,j,:,:] = np.transpose(transform(np.transpose(X_train[i,j,:,:])))
            for i in range(X_test.shape[0]):
                X_test[i,j,:,:] = np.transpose(transform(np.transpose(X_test[i,j,:,:])))
    
    return X_train, X_test


def balancing_A1(X_train, y_train, ratio="base", balancing_method="duplicate/remove"):

    y_train=np.array(y_train)

    if ratio == "50/50": #on suppose que nb_f < nb_h (dans le cas contraire il suffirait d'inverser nb_f et nb_h, ainsi que mask_f et mask_
        
        mask_h = np.where(y_train==0)[0] #indices de tous les hommes
        mask_f = np.where(y_train==1)[0] #indices de toutes les femmes
        nb_h = len(mask_h) #nombre d'hommes
        nb_f = len(mask_f) #nombre de femmes

        if 'SMOTE' in balancing_method:
            if balancing_method == 'rem/SMOTE':
                mask_h = mask_h[:-(2*nb_f)] #liste de taille (nb_h - (2*nb_f)) d'indices d'hommes à supprimer
                X_train = np.delete(X_train, mask_h, 0) #suppression de la liste des hommes
                y_train = np.delete(y_train, mask_h, 0)
            X, y = SMOTE(sampling_strategy=1, n_jobs=-2).fit_resample(X_train[:,1,:], y_train)
            X = np.zeros((X.shape[0], X.shape[1], 500))
            for i in range(X.shape[1]):
                X[:,i,:], y = SMOTE(sampling_strategy=1, n_jobs=-2).fit_resample(X_train[:,i,:], y_train)
            X_train = X
            y_train = y
        
        else:

            if balancing_method == "remove":
                mask_h = mask_h[:-nb_f] #liste de taille (nb_h - nb_f) d'indices d'hommes ) à supprimer
                mask_f = []

            elif balancing_method == "duplicate":
                mask_h = []
                mask_f = np.resize(mask_f, nb_h - nb_f) #liste de taille (nb_h - nb_f) d'indices de femmes à rajouter (potentiellement répétés)

            else: # duplicate/remove
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
    print("L'échantillon de training comporte " + str(y_train.shape[0]) + ' frames')

    return X_train, y_train, prop_HF


def datatreat_A1(x0, y0, x_final=None, train_size=0.8, Shuffle=True, preprocess=None, ratio='base', balancing_method='duplicate/remove'):
    
    if x_final is None:
        x_train, x_test, y_train, y_test = train_test_split(x0, y0, train_size=train_size, shuffle=Shuffle)
    else:
        x_train = x0
        y_train = y0
        x_test = x_final

    x_train, x_test = preprocess_A1(x_train, x_test, preprocess)

    x_train=np.concatenate(x_train, axis=0)  #sépération des 40 fenêtres indépendantes (comme si chaque fenêtre correspondait à une personne)
    y_train=np.repeat(y_train, 40)  #Multiplication par 40 de chaque personne (car séparation des fenêtres)
    x_train, y_train = shuffle(x_train, y_train)

    x_train, y_train, prop_HF = balancing_A1(x_train, y_train, ratio, balancing_method) # équilibrage homme-femme dans le dataset
    x_train, y_train = shuffle(x_train, y_train)

    x_train = x_train.reshape(x_train.shape[0], 7, 500, 1)  #Ajout d'une dimension aux données

    if x_final is None:
        return x_train, x_test, y_train, y_test, prop_HF
    else:
        return x_train, x_test, y_train, prop_HF


def datatreat_J1(X0, y0, train_size=0.8, Shuffle=True, preprocess=None, ratio='base', balancing_method='duplicate/remove'):
    x_train, x_test, y_train, y_test = train_test_split(X0, y0, train_size=train_size, shuffle=Shuffle)

    x_train, x_test = preprocess_A1(x_train, x_test, preprocess)

    x_train = np.concatenate(x_train, axis=0)  #sépération des 40 fenêtres indépendantes (comme si chaque fenêtre correspondait à une personne)
    y_train = np.repeat(y_train, 40)  #Multiplication par 40 de chaque personne (car séparation des fenêtres)
    x_train, y_train = shuffle(x_train, y_train)

    x_train, y_train, prop_HF = balancing_A1(x_train, y_train, ratio, balancing_method) # équilibrage homme-femme dans le dataset
    x_train, y_train = shuffle(x_train, y_train)

    x_train=x_train.transpose(0,2,1)  #Echange des 2èmes et 3èmes dimensions (dimension canal de taille 7 et dimension EEG de taille 500)
    x_test=x_test.transpose(0,1,3,2)

    return x_train, x_test, y_train, y_test, prop_HF

def datatreat_J2(X0, y0, train_size=0.8, Shuffle=True, preprocess=None, ratio='base', balancing_method='duplicate/remove'):
    x_train, x_test, y_train, y_test = train_test_split(X0, y0, train_size=train_size, shuffle=Shuffle)

    x_train, x_test = preprocess_A1(x_train, x_test, preprocess)

    x_train = np.concatenate(x_train.transpose(1,2,0,3)).transpose(1,0,2)  #regroupement des 40 fenêtres indépendantes comme des canaux supplémentaires (40*7 canaux par personne)
    x_test = np.concatenate(x_test.transpose(1,2,0,3)).transpose(1,0,2)

    x_train, y_train, prop_HF = balancing_A1(x_train, y_train, ratio, balancing_method) # équilibrage homme-femme dans le dataset
    x_train, y_train = shuffle(x_train, y_train)

    x_train=x_train.transpose(0,2,1)  #Echange des 2èmes et 3èmes dimensions (dimension canal de taille 7*40 et dimension EEG de taille 500)
    x_test=x_test.transpose(0,2,1)

    return x_train, x_test, y_train, y_test, prop_HF


def datatreat_J3(X0, y0, train_size=0.8, seq_len=100, pas=25, Shuffle=True, preprocess=None, ratio='base', balancing_method='duplicate/remove'):
    x_train, x_test, y_train, y_test = train_test_split(X0, y0, train_size=train_size, shuffle=Shuffle)

    x_train, x_test = preprocess_A1(x_train, x_test, preprocess)

    x_train = np.concatenate(x_train, axis=0)  #sépération des 40 fenêtres indépendantes (comme si chaque fenêtre correspondait à une personne)
    y_train = np.repeat(y_train, 40)  #Multiplication par 40 de chaque personne (car séparation des fenêtres)
    x_train, y_train = shuffle(x_train, y_train)

    x_train, y_train, prop_HF = balancing_A1(x_train, y_train, ratio, balancing_method) # équilibrage homme-femme dans le dataset
    x_train, y_train = shuffle(x_train, y_train)

    x_train=x_train.transpose(0,2,1)  #Echange des 2èmes et 3èmes dimensions (dimension canal de taille 7 et dimension EEG de taille 500)
    x_test=x_test.transpose(0,1,3,2)

    N=int((x_train.shape[1] - seq_len)/pas) #cacul du nombre de sous séquence de longueur seq_len qu'on va extraire avec un pas de longueur pas
    
    x_train_decoupe = np.zeros((N*len(x_train), seq_len, 7))
    for i in range(len(x_train)): #découpage de chaque élément de x_train en N séquences de longueur seq_len, et ajout de toutes ses séquences dans une array de longueur N*len(x_train)
        for j in range(N):
            x_train_decoupe[i*j]=x_train[i, j*pas:j*pas+seq_len, :]
    y_train_decoupe = np.repeat(y_train, N) #multiplication par N de chaque personne (car N séquences correspondent à 1 personne)

    x_train, y_train = shuffle(x_train_decoupe, y_train_decoupe)

    return x_train, x_test, y_train, y_test, prop_HF