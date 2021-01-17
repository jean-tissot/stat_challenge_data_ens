# Challenge data - Cours de STAT - Centrale Nantes
## Structure du projet
Pour le moment projet sous forme de fichiers .py, à mettre à la fin sous forme d'un Jupiter Notebook  
- Un fichier pour **charger les données** (read_data.py)  
- Un fichier pour **traiter les données** (treat_data.py)  
- Un fichier pour **créer le(s) modèle(s)** (model.py)  
- Un fichier pour **entrainer le(s) modèle(s)** (train_model.py)  
- Un fichier pour **tester le(s) modèle(s)** (test_model.py)  
Dans chaque fichier chacun crée sa fonction en lui donnant un nom logique (sauf s'il compte utiliser une fonction faite par un autre). Par exemple dans model.py si qqn utilise un *modèle 1* et qu'un autre utilise un *modèle 2*, il devrait y avoir deux fonctions (*model1* et *model2*) prenant en paramètres ceux dont on a besoin et retournant un modèle.

## But du projet
Etre capable de prédire le sexe d'une personne à partir de son activité cérébrale la nuit.

### Données
Pour chaque **donnée** *x* (activité cérébrale) correspond la **classe** *y* (sexe) correspondante

#### Donnée x
40 segments indépendant de 2 secondes de 7 canaux d'EEG (activité cérébrale) à 250 Hz  
1 canal est donc constitué de 2*250 = 500 valeurs  
Chaque donnée est donc de taille (40, 7, 500)

#### Donnée y
0: sexe masculin  
1: sexe féminin

## Méthode de résolution
Les données d'entrées sont des données temporelles (activité cérébrale de la personne pendant la nuit)  
La donnée de sortie est une classification binaire (sexe de la personne) 

#### Chargement des données (fichier read_data.py)
x: données au format h5  
y: données au format csv

#### Traitement des données (fichier treat_data.py)
Est-ce que pour une personne toutes les données sont utiles (sélectionner les données les plus représentatives)  
Les données sont temporelles. Transformation de Fourrier ?

#### Choix du modèle (fichier deep_learning_lstm.py (ou autre_methode.py))
Machine learning ? Deep learning ?  
Modèle qui tient en compte le fait que ce sont des données temporelles
Utilisation d'un réseau de neuronnes récurrents ? (modele LSTM ?)  
https://www.margo-group.com/fr/actualite/tutoriel-quelques-bases-python-prediction-de-series-temporelles/  
https://ichi.pro/fr/classification-des-series-temporelles-pour-la-reconnaissance-de-l-activite-humaine-avec-les-lstm-utilisant-tensorflow-2--202405754633727

#### Apprentissage
Pouvoir mettre en pause l'apprentissage ?  
Pouvoir sauvegarder le modèle

