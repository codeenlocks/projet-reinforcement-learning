#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'entraînement général pour le Q-Learning Tabulaire (Taquin NxN).
Inclut le suivi des performances (RAM, Vitesse) et des sauvegardes régulières.
"""

from taquin_env import TaquinEnv
from ai_models import QLearningAI
from benchmark_logger import BenchmarkLogger

# =========================================================================
# CONFIGURATION GÉNÉRALE
# =========================================================================
N = 3                        # Valeur modifiable (3 pour entraînement, 4 pour test de limite)
NUM_EPISODES = 200_000       # Nombre total d'épisodes prévus
MAX_STEPS_PER_EPISODE = 2000 # Limite de pas pour éviter de tourner en boucle
LOG_INTERVAL = 1000          # Écriture dans le CSV tous les X épisodes
SAVE_INTERVAL = 10_000       # Sauvegarde de la Q-Table tous les X épisodes

# =========================================================================
# INITIALISATION
# =========================================================================
# 1. Environnement
env = TaquinEnv(n=N)
env.load_bank_if_needed()

# 2. Agent IA
agent = QLearningAI(
    n=N, 
    exploration_rate=1.0, 
    min_exploration_rate=0.01, 
    exploration_decay=0.99995, 
    learning_rate=0.5,
    min_learning_rate=0.01,
    learning_decay=0.99999
)

# 3. Logger système
logger = BenchmarkLogger(filename=f"training_metrics_{N}x{N}.csv")

print("==================================================")
print(f" DÉBUT DE L'ENTRAÎNEMENT - TAQUIN {N}x{N}")
print(f" Objectif : {NUM_EPISODES} épisodes")
print("==================================================")

cumulative_steps = 0

# =========================================================================
# BOUCLE D'APPRENTISSAGE
# =========================================================================
for episode in range(1, NUM_EPISODES + 1):
    # Démarrage rapide depuis un état aléatoire de la banque (.npy)
    state = env.reset(path=0)
    
    done = False
    steps = 0

    while not done and steps < MAX_STEPS_PER_EPISODE:
        valid_actions = env.get_valid_actions()
        action = agent.choose_action(state, valid_actions)

        next_state, reward, done = env.step(action)
        next_valid_actions = env.get_valid_actions()

        # Mise à jour de la Q-Table (Équation de Bellman)
        agent.learn(state, action, reward, next_state, next_valid_actions, done)

        state = next_state
        steps += 1
        cumulative_steps += 1

    # Mise à jour des hyperparamètres (baisse progressive d'Epsilon et Alpha)
    agent.update_hyperparameters()

    # =====================================================================
    # LOGS ET SAUVEGARDES
    # =====================================================================
    
    # 1. Suivi des performances
    if episode % LOG_INTERVAL == 0:
        ram_mb, steps_per_sec = logger.log(episode, len(agent.Q), cumulative_steps)
        
        print(f"Ep {episode:<7} | "
              f"RAM: {ram_mb:>6.1f} Mo | "
              f"Vit: {steps_per_sec:>6.0f} st/s | "
              f"Q-Table: {len(agent.Q):>8} états | "
              f"Eps: {agent.eps:.4f}")
        
        cumulative_steps = 0 # Reset pour mesurer la vitesse du prochain bloc

    # 2. Sauvegarde de sécurité (Checkpoint)
    if episode % SAVE_INTERVAL == 0:
        # Enregistre une version spécifique (ex: v10000, v20000...)
        agent.save_q_table(version=episode)
        print(f"--- 💾 Checkpoint : Q-Table sauvegardée à l'épisode {episode} ---")

# =========================================================================
# FIN D'ENTRAÎNEMENT
# =========================================================================
print("\n[FIN] Entraînement terminé avec succès !")
agent.save_q_table(version="final")
agent.save_q_table_json(version="final")