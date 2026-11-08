#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 21:49:11 2026

@author: codeenlocks
"""

"""
Module de suivi automatique des métriques système pour le Q-Learning.
"""

import os
import time
import csv
import psutil


class BenchmarkLogger:
    def __init__(self, filename="crash_test_metrics.csv"):
        self.filename = filename
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        self.last_step_time = self.start_time
        
        # Initalisation du CSV avec en-têtes
        with open(self.filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", 
                "q_table_states", 
                "ram_usage_mb", 
                "steps_per_sec", 
                "elapsed_time_sec"
            ])

    def log(self, episode, q_table_size, steps_in_interval=1000):
        current_time = time.time()
        elapsed = current_time - self.start_time
        interval_time = current_time - self.last_step_time
        self.last_step_time = current_time
        
        # Calcul de la vitesse d'exécution
        steps_per_sec = steps_in_interval / interval_time if interval_time > 0 else 0
        
        # Récupération de la consommation mémoire exacte du processus Python (en Mo)
        ram_mb = self.process.memory_info().rss / (1024 * 1024)
        
        # Écriture dans le CSV
        with open(self.filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, 
                q_table_size, 
                round(ram_mb, 2), 
                round(steps_per_sec, 2), 
                round(elapsed, 2)
            ])
        
        return ram_mb, steps_per_sec