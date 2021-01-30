#from data import dataread, datatreat_4
import os.path
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import itertools
import matplotlib.pyplot as plt


def vector_generator(data):
    #génère un vecteur de dimension 1 à partir d'un vecteur de dimension 2 ou plus
    #au lieu de retourner une liste contenant toutes les données (ce qui prend trop de temps) on retourne élément par élément à la demande
    for d in data:
        try:
            d.shape #si génère une erreur c'est que d est nombre et qu'il faut donc directement le retourner (ce qu'on fait dans le except)
            for d_recurrent in vector_generator(d):
                yield d_recurrent
        except:
            yield d


def print_load(pourcentage, message=""):
    sys.stdout.write("\r" + message +" {:.2f} %".format(100*pourcentage))


def save_model(model, id):
    if os.path.isfile('models/'+id)==False:
        model.save('models/'+id)
    else:
        print('Modèle déjà enregistré')


def load_model(id):
    return keras.models.load_model('models/'+id)


def save_results(id, treat, model, epochs, batch_size, accuracy, roc, f1_macro, f1_wei, valid=0):
    if os.path.isfile('results/'+id+'_R.txt')==False:
        file=open('results/'+id+'_R.txt', 'w+')
        file.write("Acquisition des des donnees -> "+ treat + "\n")
        file.write("\n")
        file.write("Modele -> " + model + "\n")
        file.write("\n")
        file.write("Parametres d'apprentissage :\n")
        file.write("\n")
        file.write("Epochs -> "+str(epochs)+"\n")
        file.write("Batch_size -> "+str(batch_size)+"\n")
        file.write("Validation -> "+str(valid)+"\n")
        file.write("\n")
        file.write("Resultats du test :\n")
        file.write("\n")
        file.write("Accuracy -> "+str(accuracy)+"\n")
        file.write("ROC AUC -> "+str(roc)+"\n")
        file.write("f1 score (macro) -> "+str(f1_macro)+"\n")
        file.write("f1 score (weighted) -> "+str(f1_wei)+"\n")
        file.close()
    else:
        print('Résultats déjà enregistré')


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, id=''):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('results/'+id+'_CM.png')


def plot_loss_acc_history(history, id, validation_split=0):
    loss_train=history.history['loss']
    acc_train=history.history['accuracy']
    if validation_split>0:
        acc_val=history.history['val_accuracy']
        loss_val=history.history['val_loss']
    
    x=range(1,len(loss_train)+1,1)

    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle('Evolution of loss and accuracy during the training')
    
    axs[0].plot(x, loss_train, linewidth=2, color='red', label="Training")
    if validation_split>0: 
        axs[0].plot(x, loss_val, linewidth=2, color='blue', label="Validation")
    axs[0].set_title('Loss')
    axs[0].legend(shadow=True, fancybox=True)

    axs[1].plot(x, acc_train, linewidth=2, color='red', label="Training")
    if validation_split>0:
        axs[1].plot(x, acc_val, linewidth=2, color='blue', label="Validation")
    axs[1].set_title('Accuracy')
    axs[1].legend(shadow=True, fancybox=True)

    fig.savefig('results/'+id+'_HIST.png')


# Fonctions d'activation custom
def square_nonlin(x):
    orig = x
    x = tf.where(orig >2.0, (tf.ones_like(x)) , x)
    x = tf.where(tf.logical_and(0.0 <= orig, orig <=2.0), (x - tf.math.square(x)/4.), x)
    x = tf.where(tf.logical_and(-2.0 <= orig, orig < 0), (x + tf.math.square(x)/4.), x)
    return tf.where(orig < -2.0, -tf.ones_like(x), x)

def square(x):
  return x*x

def log(x, eps=1e-6):
    #eviter log(0) en imposant une valeur min eps
    return tf.math.log(tf.clip_by_value(x, clip_value_min=eps))