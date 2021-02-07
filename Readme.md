# Challenge data - Cours de STAT - Centrale Nantes
Url du challenge: https://challengedata.ens.fr/participants/challenges/27/

## Structure du projet
Projet python sous forme de fichiers .py :
- Un fichier pour **charger et traiter les données** (data.py)  
- Un fichier pour **créer les modèles** (models.py)  
- Un fichier pour **tester les modèles** (test.py)  
- Un fichier pour **définir des fonctions utiles** (tools.py)  
- Un fichier pour **superviser l'appel des différentes fonctions** (main_X.py ou notebook_X.ipynb)  
 
Dans chaque fichier on crée ses fonction en leur donnant des noms logique (sauf si on compte utiliser une fonction faite par un autre). Par exemple dans models.py si qqn utilise un *modèle 1* et qu'un autre utilise un *modèle 2*, il devrait y avoir deux fonctions (*model1* et *model2*) prenant en paramètres ceux dont on a besoin et retournant un modèle. 

Cet ensemble de fichiers nous a servi pour nos différents tests afin de déterminer quel modèle retenir pour la classification des données du challenge.  
**Les lignes de codes de cette classification ont été regroupées dans le notebook *projet.ipynb* .**

## But du projet
Etre capable de prédire le sexe d'une personne à partir de son activité cérébrale la nuit.  

### Données
Pour chaque **donnée** *x* (activité cérébrale) correspond la **classe** *y* (sexe) correspondante

#### Donnée x
40 segments indépendant de 2 secondes de 7 canaux d'EEG (activité cérébrale) à 250 Hz  
1 canal est donc constitué de 2*250 = 500 valeurs  
Chaque donnée est donc de taille (40, 7, 500)  
  
![représentation data](https://user-images.githubusercontent.com/77540676/104845639-c2d64500-58d6-11eb-9d62-2e5f522b062f.JPG)  

#### Donnée y
0: sexe masculin  
1: sexe féminin  

## Méthode de résolution
Les données d'entrées sont des données temporelles (activité cérébrale de la personne pendant la nuit)  
La donnée de sortie est une classification binaire (sexe de la personne)  

### Chargement des données (fichier data.py)
x: données au format h5  
y: données au format csv  

### Traitement des données (fichier treat_data.py)
Est-ce que pour une personne toutes les données sont utiles (sélectionner les données les plus représentatives)   
Les données sont temporelles. Transformation de Fourrier ?  

### Choix du modèle (fichier models.py)
Machine learning ? Deep learning ?  
Modèle qui tient en compte le fait que ce sont des données temporelles  
Utilisation d'un réseau de neuronnes récurrents ? (modele LSTM ?)  
https://www.margo-group.com/fr/actualite/tutoriel-quelques-bases-python-prediction-de-series-temporelles/  
https://ichi.pro/fr/classification-des-series-temporelles-pour-la-reconnaissance-de-l-activite-humaine-avec-les-lstm-utilisant-tensorflow-2--202405754633727  

Tests de deux grandes familles de modèles:
- Les CNN (Convolutional Neural Network)
- Les LSTM (Long Short-Term Memory)

### Apprentissage
Pouvoir mettre en pause l'apprentissage  
Pouvoir sauvegarder le modèle  

