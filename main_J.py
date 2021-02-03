from data import dataread, datatreat_A1, datatreat_J1
from test import test_1
from models import model_of_test, lstm_model
from tools import plot_loss_acc_history

treat_function = datatreat_J1
my_model = lstm_model
test_fonction = test_2
epochs=50
batch_size=100
id='test'
preprocess='None'

print("loading data...")
x, y, x_final = dataread()

print("treating data...")
x_train, x_test, y_train, y_test, prop = treat_function(x, y, preprocess=preprocess, balancing_method='duplicate', ratio='50/50')

print("\nforme des données d'entrée: ", x_train[0].shape)
model = my_model(x_train[0].shape)
model.summary()
print("\ntraining model...")
history=model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

print("testing model...")
accuracy, roc, f1_macro, f1_wei = test_fonction(model, x_test, y_test, id)

#print("results (accuracy, roc, f1_macro, f1_wei):", accuracy, roc, f1_macro, f1_wei)
plot_loss_acc_history(history, id, 0.1)