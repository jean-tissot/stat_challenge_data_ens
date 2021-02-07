import h5py,pandas as pd, csv
import matplotlib.pyplot as plt, numpy as np
import os.path,sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from tensorflow import keras
from tensorflow.keras import layers

#Acquisition des données
X_file=h5py.File("data/X_train_new.h5",'r') # X est ici au format "HDF5 dataset"
X_train=np.array(X_file['features']) # X est ici au format "ndarray"

X_final_file=h5py.File("data/X_test_new.h5",'r') # X est ici au format "HDF5 dataset"
X_test=np.array(X_final_file['features']) # X est ici au "ndarray"

y=pd.read_csv("data/y_train_AvCsavx.csv") # y est ici au format "pandas DataFrame"
y_train=np.array(y['label']) # y est ici au format "ndarray"

#préprocessing des données
transform = StandardScaler().fit_transform #fonction de standardisation
for j in range(40):
    for i in range(X_train.shape[0]):
        X_train[i,j,:,:] = np.transpose(transform(np.transpose(X_train[i,j,:,:])))
    for i in range(X_test.shape[0]):
        X_test[i,j,:,:] = np.transpose(transform(np.transpose(X_test[i,j,:,:])))

X_train=np.concatenate(X_train, axis=0)  #sépération des 40 fenêtres indépendantes (comme si chaque fenêtre correspondait à une personne)
y_train=np.repeat(y_train, 40)  #Multiplication par 40 de chaque personne (car séparation des fenêtres)
X_train, y_train = shuffle(X_train, y_train) #mélange des données pour ne pas avoir 40 fois le même sexe d'affilé

#rééquilibrage du dataset de training
mask_h = np.where(y_train==0)[0] #indices de tous les hommes
mask_f = np.where(y_train==1)[0] #indices de toutes les femmes
nb_h = len(mask_h) #nombre d'hommes
nb_f = len(mask_f) #nombre de femmes
nb = (nb_h - nb_f)//2
mask_h = mask_h[:nb] #liste de taille (nb_h - nb_f)/2 d'indices d'hommes ) à supprimer
mask_f = np.resize(mask_f, nb) #liste de taille (nb_h - nb_f)/2 d'indices de femmes à rajouter (potentiellement répétés)

add_X = X_train[mask_f,...] #liste des femmes à rajouter
add_y = y_train[mask_f]
X_train = np.delete(X_train, mask_h, 0) #suppression de la liste des hommes
y_train = np.delete(y_train, mask_h, 0)
X_train=np.concatenate((X_train, add_X), axis=0) #ajout de la liste de femmes
y_train=np.concatenate((y_train, add_y), axis=0)

X_train, y_train = shuffle(X_train, y_train) #mélange des données synthétiques aux données existantes

X_train = X_train.reshape(X_train.shape[0], 7, 500, 1)  #Ajout d'une dimension aux données pour entrer dans le CNN

#création du CNN
model = keras.Sequential(
[
    layers.Conv2D(filters=25, kernel_size=(1,10), strides=1, padding='same', input_shape=(7,500,1)),
    layers.Conv2D(filters=25, kernel_size=(7,1), strides=1, padding='same', use_bias=False),
    layers.BatchNormalization(momentum=0.1, epsilon=0.00001),
    layers.Activation('elu'),
    layers.MaxPool2D(pool_size=(1,3), strides=(1,3), padding='same'),
    
    layers.Dropout(0.5),
    layers.Conv2D(filters=50, kernel_size=(1,10), strides=1, padding='same', use_bias=False),
    layers.BatchNormalization(momentum=0.1, epsilon=0.00001),
    layers.Activation('elu'),
    layers.MaxPool2D(pool_size=(1,3), strides=(1,3), padding='same'),

    layers.Dropout(0.5),
    layers.Conv2D(filters=100, kernel_size=(1,10), strides=1, padding='same', use_bias=False),
    layers.BatchNormalization(momentum=0.1, epsilon=0.00001),
    layers.Activation('elu'),
    layers.MaxPool2D(pool_size=(1,3), strides=(1,3), padding='same'),

    layers.Dropout(0.5),
    layers.Conv2D(filters=200, kernel_size=(1,10), strides=1, padding='same', use_bias=False),
    layers.BatchNormalization(momentum=0.1, epsilon=0.00001),
    layers.Activation('elu'),
    layers.MaxPool2D(pool_size=(1,3), strides=(1,3), padding='same'),

    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
]
)
model.compile(
loss='binary_crossentropy',
optimizer=keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
metrics=[keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.AUC(name='AUC')]
)
model.summary()

#entrainement du modèle
epochs=2
batch_size=69
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

#application du modèle aux données à classifier
y_pred=[]
for i in range(40):
    X_test_i=X_test[:,i,:,:]
    X_test_i=X_test_i.reshape(X_test_i.shape[0], X_test_i.shape[1], X_test_i.shape[2], 1)
    y_pred.append(model.predict(X_test_i))
y_pred=np.mean(y_pred, axis=0)

#export de la classification au format csv
with open('classification.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "label"])
    for i in range(len(y_pred)):
        writer.writerow([i, int(y_pred[i]>0.5)])