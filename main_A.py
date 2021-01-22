from read_data import dataread
from treat_data import datatreat_1, datatreat_2
import numpy as np


X, y, X_final = dataread()

X_train, X_test, y_train, y_test = datatreat_1(X, y)

print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(y_train))
print(np.shape(y_test))