# Challenge data - Cours de STAT - Centrale Nantes
## Format du projet
Pour le moment le projet est réduit à un fichier .py, mais il devrait ensuite être sous la forme d'un jupiter notebook
## But du projet
Etre capable de prédire le sexe d'une personne à partir de son activité cérébrale la nuit.
### Données
Pour chaque **donnée** *x* correspond la **classe** *y* (sexe) correspondante
#### Donnée x
40 segments indépendant de 2 secondes de 7 canaux d'EEG (activité cérébrale) à 250 Hz
*1 canal:* est donc constitué de 2*250 = 500 valeurs
Chaque donnée est donc de taille (40, 7, 500)
#### Donnée y
0: sexe masculin
1: sexe féminin

## Méthode de résolution
Les données d'entrées sont des données temporelles (activité cérébrale de la personne pendant la nuit)
La donnée de sortie est une classification binaire (sexe de la personne)
Utilisation d'un réseau de neuronnes récurrents (modele LSTM ?)