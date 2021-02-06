from tools import plot_confusion_matrix, plot_loss_acc_history
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score


def test(predicts, y_test, id):
    predicts2=np.transpose(predicts>0.5)[0]
    accuracy=(np.count_nonzero(predicts2==y_test))/len(y_test)*100
    roc=roc_auc_score(y_test, predicts2)
    f1_macro=f1_score(y_test, predicts2, average='macro')
    f1_wei=f1_score(y_test, predicts2, average='weighted')

    print('')
    print('Accuracy : ' + str(accuracy))
    print('')
    print('ROC AUC : ' + str(roc))
    print('')
    print('f1 score (macro) : ' + str(f1_macro))
    print('')
    print('f1 score (weighted) : ' + str(f1_wei))
    
    cm = confusion_matrix(y_true=y_test, y_pred=predicts2)
    cm_labels = ['male','female']
    plot_confusion_matrix(cm=cm, classes=cm_labels, title='Confusion Matrix', id=id)

    return accuracy, roc, f1_macro, f1_wei


def test_1(model, X_test, y_test=None, id=None, increase_dim=True):
    predicts=[]
    for i in range(40):
        X_test_i=X_test[:,i,:,:]
        if increase_dim:
            X_test_i=X_test_i.reshape(X_test_i.shape[0], X_test_i.shape[1], X_test_i.shape[2], 1)
        predicts.append(model.predict(X_test_i))
    predicts=np.mean(predicts, axis=0)

    if y_test:
        return test(predicts, y_test, id)
    else:
        return predicts


def test_J1(model, x_test, y_test, id):
    return test_1(model, x_test, y_test, id, increase_dim=False)

def test_J2(model, X_test, y_test, id):
    predicts = model.predict(X_test)
    return test(predicts, y_test, id)

def test_J3(model, X_test, y_test, id, seq_len=100, pas=25):
    N=int((X_test.shape[2] - seq_len)/pas)
    predicts=np.zeros((len(X_test), 40, N, 1))
    for k in range(40):
        X_test_k = X_test[:,k,:,:]
        for j in range(N):
            predicts[:,k,j,:]=model.predict(X_test_k[:, j*pas:j*pas+seq_len, :])
    predicts=np.mean(np.mean(predicts,axis=2),axis=1)
    return test(predicts, y_test, id)
