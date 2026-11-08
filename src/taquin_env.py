#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'implémentation optimisée de la logique du jeu de Taquin
"""

import os
import random
import numpy as np


class TaquinEnv:
    def __init__(self, n=2, path=None):
        self.n = n
        self.current_manhattan = 0
        self.bank = None
        self.reset(path=path)

    # =========================================================================
    # OPTIMISATIONS MATHÉMATIQUES (Vectorisation & O(1))
    # =========================================================================

    def calcule_signature(self, plate):
        """ Signature Sigma(epsilon) vectorisée en 1D. """
        flat = plate.flatten()
        tiles = flat[flat != 0]
        
        # Comparaison vectorisée de toutes les paires i < j
        i_idx, j_idx = np.triu_indices(len(tiles), k=1)
        inversions = np.sum(tiles[i_idx] > tiles[j_idx])
        
        r0 = np.argwhere(plate == 0)[0][0] + 1
        delta = 1 if (self.n % 2 == 0) else 0
        
        return (inversions + delta * (self.n - r0)) % 2

    def is_solvable(self, plate):
        resolved = np.arange(1, self.n * self.n + 1)
        resolved[-1] = 0
        return self.calcule_signature(plate) == self.calcule_signature(resolved.reshape((self.n, self.n)))

    def get_single_manhattan(self, val, r, c):
        """ Calcule la distance de Manhattan pour une seule tuile. """
        if val == 0: return 0
        target_r = (val - 1) // self.n
        target_c = (val - 1) % self.n
        return abs(r - target_r) + abs(c - target_c)

    def _get_total_manhattan(self):
        dist = 0
        for r in range(self.n):
            for c in range(self.n):
                dist += self.get_single_manhattan(self.plate[r, c], r, c)
        return dist

    # =========================================================================
    # GESTION BANQUE Binaire (.npy)
    # =========================================================================

    def generate_and_save_bank(self, num_plates=50000):
        """ Génère rapidement une banque soluble via marches aléatoires et sauvegarde en .npy. """
        folder = os.path.join("plates", f"{self.n}x{self.n}")
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, "bank.npy")
        
        print(f"[INIT] Génération de {num_plates} plateaux solubles...")
        
        solved = np.arange(1, self.n * self.n + 1)
        solved[-1] = 0
        solved_plate = solved.reshape((self.n, self.n))
        
        bank_arrays = []
        for _ in range(num_plates):
            plate = solved_plate.copy()
            # Marche aléatoire de 150 coups (garantit la solubilité par construction)
            for _ in range(150):
                r0, c0 = np.argwhere(plate == 0)[0]
                moves = []
                if r0 > 0: moves.append((-1, 0))
                if r0 < self.n - 1: moves.append((1, 0))
                if c0 > 0: moves.append((0, -1))
                if c0 < self.n - 1: moves.append((0, 1))
                
                dr, dc = random.choice(moves)
                plate[r0, c0], plate[r0+dr, c0+dc] = plate[r0+dr, c0+dc], plate[r0, c0]
            bank_arrays.append(plate)
            
        np.save(filepath, np.array(bank_arrays))
        print(f"[SUCCÈS] Banque sauvegardée dans {filepath}")

    def load_bank_if_needed(self):
        if self.bank is None:
            filepath = os.path.join("plates", f"{self.n}x{self.n}", "bank.npy")
            if os.path.exists(filepath):
                self.bank = np.load(filepath)
            else:
                self.generate_and_save_bank()
                self.bank = np.load(filepath)

    # =========================================================================
    # REINITIALISATION ET BOUCLE DU JEU
    # =========================================================================

    def reset(self, path=None):
        if path == 0:
            self.load_bank_if_needed()
            idx = random.randint(0, len(self.bank) - 1)
            self.plate = self.bank[idx].copy()
        else:
            # Génération par défaut à la volée
            arr = np.arange(self.n * self.n)
            while True:
                np.random.shuffle(arr)
                candidate = arr.reshape((self.n, self.n))
                if self.is_solvable(candidate):
                    self.plate = candidate
                    break
                    
        self.current_manhattan = self._get_total_manhattan()
        return self.plate.copy()

    def get_valid_actions(self):
        r0, c0 = np.argwhere(self.plate == 0)[0]
        actions = []
        if r0 > 0: actions.append('U')
        if r0 < self.n - 1: actions.append('D')
        if c0 > 0: actions.append('L')
        if c0 < self.n - 1: actions.append('R')
        return actions

    def step(self, action):
        r0, c0 = np.argwhere(self.plate == 0)[0]
        moves = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        dr, dc = moves[action]
        nr, nc = r0 + dr, c0 + dc

        # 1. Calcul de la distance de la tuile AVANT le swap
        moving_tile = self.plate[nr, nc]
        old_dist = self.get_single_manhattan(moving_tile, nr, nc)

        # 2. Swap (Déplacement)
        self.plate[r0, c0], self.plate[nr, nc] = self.plate[nr, nc], self.plate[r0, c0]

        # 3. Calcul de la distance APRES le swap et mise à jour différentielle (O(1))
        new_dist = self.get_single_manhattan(moving_tile, r0, c0)
        delta_h = old_dist - new_dist  # Positif si on s'est rapproché
        self.current_manhattan -= delta_h

        # Condition de victoire : Manhattan == 0 et la case vide est à la fin
        done = (self.current_manhattan == 0 and self.plate[-1, -1] == 0)
        
        # Récompense
        reward = 100.0 if done else (-1.0 + delta_h * 2.0)

        return self.plate.copy(), reward, done

    def render(self):
        print("+" + "---+" * self.n)
        for row in self.plate:
            row_str = "|"
            for val in row:
                row_str += "   |" if val == 0 else f" {val:2d}|"
            print(row_str)
            print("+" + "---+" * self.n)
        print()