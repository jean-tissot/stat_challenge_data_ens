from data import dataread, datatreat_1, datatreat_2, datatreat_3, datatreat_4
from test import test_1
from models import model_of_test

treat_function=datatreat_3
my_model=model_of_test
test_fonction = test_1
epochs=20
batch_size=32
id='test'

print("loading data...")
x, y, x_final = dataread()

print("treating data...")
x_train, x_test, y_train, y_test = treat_function(x, y)

print("training model...")
model = my_model(x_train[0].shape)
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

print("testing model...")
accuracy, roc, f1_macro, f1_wei = test_1(model, x_test, y_test, id)

print("results (accuracy, roc, f1_macro, f1_wei):", accuracy, roc, f1_macro, f1_wei)
