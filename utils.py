
import sys

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