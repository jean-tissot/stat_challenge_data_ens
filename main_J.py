from data import dataread, datatreat_A1, datatreat_J1
from test import test_1
from models import model_of_test, lstm_model

treat_function = datatreat_J1
my_model = lstm_model
test_fonction = test_1
epochs=20
batch_size=100
id='test'

print("loading data...")
x, y, x_final = dataread()

print("treating data...")
x_train, x_test, y_train, y_test, prop = treat_function(x, y, preprocess='Standardization', balancing_method='remove', ratio='50/50')

print("\nforme des données d'entrée: ", x_train[0].shape)
model = my_model(x_train[0].shape)
model.summary()
print("\ntraining model...")
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

print("testing model...")
accuracy, roc, f1_macro, f1_wei = test_1(model, x_test, y_test, id, increase_dim=False)

#print("results (accuracy, roc, f1_macro, f1_wei):", accuracy, roc, f1_macro, f1_wei)
