import numpy as np
import pandas as pd
import h5py


X_train_raw = h5py.File('C:/Users/AntoineLespinasse/Desktop/STATS/Projet/data/X_train_new.h5', 'r')

print(X_train_raw['features'].shape)

X_test_raw = h5py.File('C:/Users/AntoineLespinasse/Desktop/STATS/Projet/data/X_test_new.h5', 'r')

print(X_test_raw['features'].shape)