# Taquin AI : Résolution Algébrique et Apprentissage par Renforcement

Ce projet implémente un environnement complet pour le jeu de Taquin (15-puzzle) avec des grilles de dimension variable ($N \times N$). Il explore et compare plusieurs approches d'Intelligence Artificielle pour sa résolution, tout en démontrant empiriquement les limites physiques du Q-Learning tabulaire face au fléau de la dimension.

## Fonctionnalités Actuelles

* **Environnement Mathématiquement Rigoureux** : Basé sur le groupe des permutations $G_2$ et la théorie des invariants. La génération de plateaux garantit 100% de solubilité instantanée via des marches aléatoires depuis l'état résolu $\zeta$.
* **Optimisation Algorithmique Avancée** :
  * Calcul de la distance de Manhattan en $\mathcal{O}(1)$ (mise à jour différentielle).
  * Calcul de la signature des permutations $\Sigma(\epsilon)$ vectorisé via NumPy 1D.
  * Gestion de banques d'états ultra-rapide via des fichiers binaires `.npy`.
* **Trois Paradigmes d'IA (en 3x3 et 4x4)** :
  1. **Q-Learning Tabulaire** : Apprentissage par renforcement classique (concluant en 3x3, impossible en 4x4).
  2. **Recherche Optimale ($A^*$)** : Algorithme avec heuristique de Manhattan garantissant le chemin absolu le plus court.
  3. **Stratégie Algébrique par Réduction** : Implémentation du Lemme 2.15 (résolution de sous-problèmes ligne par ligne puis colonne par colonne jusqu'au bloc 2x2).
* **Interfaces Graphiques (Tkinter)** : Design "bois sombre", permettant de comparer en temps réel les IA et un joueur Humain.
* **Benchmarking** : Suivi des métriques système (RAM, Vitesse) pour mesurer la saturation du Q-Learning tabulaire en 4x4.

## Architecture du Projet

* `taquin_env.py` : Logique de l'environnement, validation des signatures, gestion de la banque d'états `.npy` et optimisation $\mathcal{O}(1)$.
* `ai_models.py` : Classes des modèles `QLearningAI`, `MyStrategyAI` et `AStarAI`.
* `benchmark_logger.py` : Suivi continu (psutil) de la consommation RAM et du nombre d'étapes par seconde.
* `train.py` : Script d'entraînement du Q-Learning intégrant le crash-test.
* `gui.py` : Interface de base pour la visualisation d'une IA simple.
* `gui2.py` : Interface de Benchmark 3x3 (Humain vs Q-Learning vs A* vs Stratégie Algébrique).
* `gui3.py` : Interface de Benchmark 4x4 avec panneau de Crash Test (Limites matérielles du Q-Learning).

## Le Constat : Limite du Tabulaire (Crash Test 4x4)

Le Q-Learning tabulaire stocke chaque état rencontré dans un dictionnaire (`self.Q`). 
* En **3x3** : $\approx 181\ 440$ états possibles. Faisable en quelques minutes d'entraînement.
* En **4x4** : $\approx 1,04 \times 10^{13}$ états possibles. 

**Résultat du Benchmark** : La saturation de la mémoire vive (16 Go RAM) survient après quelques millions d'états stockés. Il est matériellement impossible d'entraîner et de charger une Q-Table exhaustive pour un plateau 4x4. Ce constat justifie le passage à une approximation de fonction via les Réseaux de Neurones.

---

## Prochaines Étapes : Transition vers le Deep Q-Learning (DQN)

Pour surpasser le mur des $10^{13}$ états du Taquin 4x4, le Q-Learning tabulaire sera remplacé par un algorithme **DQN (Deep Q-Network)** (implémentation prévue avec PyTorch).

Voici les chantiers à réaliser pour cette transition :

### 1. Représentation de l'État (State Representation)
* Le réseau de neurones ne peut pas prendre un simple tuple. La grille $4 \times 4$ devra être transformée en un tenseur PyTorch.
* **Encodage One-Hot** : Représenter la grille non pas avec des entiers (0-15), mais via une matrice creuse $16 \times 4 \times 4$ (chaque tuile a son propre canal) pour aider le réseau à comprendre la spatialité sans considérer les tuiles comme des valeurs continues.

### 2. Architecture du Réseau (Le Modèle PyTorch)
* Créer une classe `DQNModel(nn.Module)`.
* En entrée : La grille encodée.
* En sortie : 4 neurones linéaires correspondant aux Q-Valeurs des 4 actions (Haut, Bas, Gauche, Droite).
* Essayer un *Multi-Layer Perceptron (MLP)* basique, puis tester des couches *Convolutionnelles (CNN)* si le MLP peine à capter la structure spatiale.

### 3. Le tampon de mémoire (Experience Replay)
* Remplacer l'apprentissage instantané de `train.py` par une mémoire circulaire (`deque` de taille $\approx 100\ 000$).
* Stocker les transitions : `(état, action, récompense, état_suivant, done)`.
* À chaque étape, échantillonner un *batch* aléatoire (ex: 64 ou 128 transitions) pour entraîner le réseau. Cela brise la corrélation temporelle des coups consécutifs.

### 4. Le Réseau Cible (Target Network)
* Utiliser la technique de la *Double Q-Learning* pour stabiliser l'apprentissage.
* Maintenir un `policy_net` (qui joue et s'entraîne) et un `target_net` (qui évalue).
* Copier les poids du `policy_net` vers le `target_net` à intervalles réguliers.

### 5. Fonction de Perte (Loss Function) et Optimiseur
* Utiliser `nn.MSELoss` ou `nn.HuberLoss` pour minimiser l'erreur entre la prédiction du réseau et la cible de Bellman.
* Utiliser un optimiseur comme Adam (`torch.optim.Adam`).
```eof
