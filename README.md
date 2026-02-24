# Projet de création de modèle prédictif, régression linéaire, afin d'estimer le prix d'une voiture

## Contexte du projet
Ce projet a pour objectif de construire un modèle de régression linéaire permettant d'estimer le prix de vente d'une voiture d'occasion. Les données proviennent d'un jeu de données relatif à des véhicules (2017) et contiennent des informations telles que l'année de fabrication, le kilométrage, le type de carburant, la transmission, le nombre de propriétaires, etc. Le cas d'usage est d'aider un acheteur, nommé Martin, à déterminer un budget en fonction de caractéristiques désirées.

## Analyse des données
Le script de notebook commence par charger les données brutes et effectuer des analyses descriptives. Après suppression des doublons (aucune valeur manquante n'est détectée), le jeu de données est enregistré dans le dossier `data/clean`.

Des visualisations sont réalisées pour mieux comprendre les distributions :
- Plusieurs histogrammes couvrant l'année, le prix de vente, le prix présent, le kilométrage, le nombre de propriétaires et le type de carburant.
- Des `catplot` avec Seaborn pour observer la variation du prix en fonction de variables catégorielles telles que le type de carburant, la transmission et le nombre de propriétaires.

Une nouvelle variable calculant l'âge du véhicule (`Age = 2017 - Year`) est ajoutée et la corrélation de Pearson entre l'âge et le prix de vente est mesurée.

## Algorithme utilisé
La méthode principale employée est la régression linéaire, abordée sous deux formes :

1. **Régression univariée**
   - Âge du véhicule en tant que variable explicative.
   - Mise en œuvre à la fois avec `scipy.stats.linregress` et `sklearn.linear_model.LinearRegression` pour obtenir pente, intercept, coefficient et score R².

2. **Régression multivariée**
   - Variables explicatives : kilométrage (`Kms_Driven`) et transmission encodée (`Transmission_enc`), où 0=manuel et 1=automatique.
   - Entraînement d'un modèle `LinearRegression` de scikit-learn, affichage des coefficients et visualisation en 3D du plan de régression sur le nuage de données.

Une fonction utilitaire `evaluate_model` calcule les métriques MSE, RMSE, MAE et R² pour chaque modèle. Les résidus sont ensuite tracés pour identifier d'éventuels problèmes.

Le notebook fournit également un exemple de filtrage des véhicules correspondant aux critères de Martin (moins de 7 ans, moins de 100 000 km, boîte manuelle) et estime leur prix moyen et médian. Le modèle multivarié est utilisé pour prédire le prix d'un véhicule typique selon ces critères.

## Conclusion
Le projet met en évidence que :
- La corrélation entre l'âge et le prix est modeste, mais suffisamment pour envisager une régression linéaire simple.
- La régression multivariée apporte une légère amélioration (R² plus élevé, MSE plus faible), mais les scores restent faibles (<0,2), indiquant que les deux caractéristiques choisies n'expliquent qu'une petite part de la variance du prix. D'autres facteurs sont nécessaires pour une modélisation robuste.
- Les résidus montrent des structures non aléatoires et des hétéroscédasticités, suggérant que la relation n'est pas entièrement linéaire ou que des variables importantes sont absentes.

En guise de bilan, ces modèles constituent un point de départ et permettent une estimation approximative des prix. Pour des prédictions plus fiables, il serait nécessaire d'intégrer davantage de variables (marque, modèle, puissance, équipement, etc.) et/ou d'explorer des algorithmes plus complexes (forêts, gradient boosting, réseaux de neurones, etc.).