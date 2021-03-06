from data import dataread, datatreat_A1, datatreat_J1, datatreat_J2, datatreat_J3
from test import test_1,test_J1, test_J2, test_J3
from models import model_of_test, lstm_model, lstm_model_2, lstm_model_3, lstm_cnn
from tools import plot_loss_acc_history, loss_generator
from tensorflow.keras.callbacks import EarlyStopping

id='cnn_stand_rem_dropout3_2_1' #'lstm_2_(14_Drop2_28_Drop2)_stand_sous-sequences_100_25'
epochs=150
batch_size=50
preprocess='standardization'
ratio='50/50'
balancing_method='remove'
validation_split=0.1
treat_function = datatreat_A1
my_model = lstm_cnn
test_fonction = test_1

print("loading data...")
x, y, x_final = dataread()

print("treating data...")
x_train, x_test, y_train, y_test, prop_HF = treat_function(x, y, preprocess=preprocess, balancing_method=balancing_method, ratio=ratio)

print("\nforme des données d'entrée: ", x_train[0].shape)
model = my_model(x_train[0].shape) #loss=loss_generator(3.5)

print("\ntraining model...")
#es=EarlyStopping(monitor='accuracy', mode='max', min_delta=0.001, patience=2, baseline=0.995, verbose=1)
history=model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

print("testing model...")
accuracy, roc, f1_macro, f1_wei = test_fonction(model, x_test, y_test, id)

#print("results (accuracy, roc, f1_macro, f1_wei):", accuracy, roc, f1_macro, f1_wei)
plot_loss_acc_history(history, id, validation_split)