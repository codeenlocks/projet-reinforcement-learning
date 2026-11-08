#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de modèles d'IA pour le jeu de taquin (Optimisé Heuristique Différentielle)
"""

import os
import pickle
import json
import ast
import random
from collections import defaultdict
import numpy as np
import heapq

# =============================================================================
# IA 1 : APPRENTISSAGE PAR RENFORCEMENT (Tabular Q-learning) - INCHANGÉ
# =============================================================================
class QLearningAI:
    def __init__(self, actions=['U', 'D', 'L', 'R'], exploration_rate=1.0, min_exploration_rate=0.01,
                 exploration_decay=0.9995, learning_rate=0.5, min_learning_rate=0.05,
                 learning_decay=0.9999, discount_factor=0.99, n=2, Q_table=None):
        self.actions = actions
        self.eps = exploration_rate
        self.min_eps = min_exploration_rate
        self.eps_decay = exploration_decay
        self.alpha = learning_rate
        self.min_alpha = min_learning_rate
        self.alpha_decay = learning_decay
        self.gamma = discount_factor
        self.n = n
        self.Q = Q_table if Q_table is not None else defaultdict(lambda: {a: 0.0 for a in self.actions})

    def get_state_key(self, state):
        if isinstance(state, np.ndarray): return tuple(map(tuple, state))
        if isinstance(state, list): return tuple(tuple(row) for row in state)
        return state

    def choose_action(self, state, valid_actions):
        state_key = self.get_state_key(state)
        if random.random() < self.eps: return random.choice(valid_actions)
        q_vals = self.Q[state_key]
        valid_q = {a: q_vals[a] for a in valid_actions}
        max_q = max(valid_q.values())
        best_actions = [a for a, q in valid_q.items() if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_valid_actions, done):
        s_key = self.get_state_key(state)
        next_s_key = self.get_state_key(next_state)
        q_predict = self.Q[s_key][action]
        q_target = reward if done else reward + self.gamma * max(self.Q[next_s_key][a] for a in next_valid_actions)
        self.Q[s_key][action] += self.alpha * (q_target - q_predict)

    def update_hyperparameters(self):
        self.eps = max(self.min_eps, self.eps * self.eps_decay)
        self.alpha = max(self.min_alpha, self.alpha * self.alpha_decay)

    def _get_filepath(self, folder, filename_base, version, ext, n=None):
        size_str = f"{n if n is not None else self.n}x{n if n is not None else self.n}"
        subfolder = os.path.join(folder, size_str)
        os.makedirs(subfolder, exist_ok=True)
        ver_str = str(version) if str(version).startswith('v') else f"v{version}" if version else ""
        filename = f"{filename_base}_{ver_str}.{ext}" if version else f"{filename_base}.{ext}"
        return os.path.join(subfolder, filename)

    def save_q_table(self, folder="backup_pkl", filename_base="q_table", version=None, n=None):
        filepath = self._get_filepath(folder, filename_base, version, "pkl", n=n)
        with open(filepath, 'wb') as f: pickle.dump(dict(self.Q), f)

    def load_q_table(self, folder="backup_pkl", filename_base="q_table", version=None, n=None):
        filepath = self._get_filepath(folder, filename_base, version, "pkl", n=n)
        if not os.path.exists(filepath): return False
        with open(filepath, 'rb') as f:
            self.Q = defaultdict(lambda: {a: 0.0 for a in self.actions}, pickle.load(f))
        return True

    def save_q_table_json(self, folder="backup_json", filename_base="q_table", version=None, n=None):
        filepath = self._get_filepath(folder, filename_base, version, "json", n=n)
        with open(filepath, 'w', encoding='utf-8') as f: json.dump({str(k): v for k, v in self.Q.items()}, f, indent=4)

    def load_q_table_json(self, folder="backup_json", filename_base="q_table", version=None, n=None):
        filepath = self._get_filepath(folder, filename_base, version, "json", n=n)
        if not os.path.exists(filepath): return False
        with open(filepath, 'r', encoding='utf-8') as f:
            self.Q = defaultdict(lambda: {a: 0.0 for a in self.actions}, {ast.literal_eval(k): v for k, v in json.load(f).items()})
        return True


# =============================================================================
# IA 2 & 3 : OPTIMISATION DES RECHERCHES A* (Calcul O(1) dans la file)
# =============================================================================
class BaseSearchAI:
    """ Classe parente contenant l'optimisation différentielle de l'heuristique """
    def __init__(self, n):
        self.n = n

    def _get_single_manhattan(self, val, r, c):
        if val == 0: return 0
        return abs(r - ((val - 1) // self.n)) + abs(c - ((val - 1) % self.n))


class MyStrategyAI(BaseSearchAI):
    def __init__(self, n=3):
        super().__init__(n)
        self.action_queue = []

    def _solve_subproblem(self, plate, locked_cells):
        solved = np.arange(1, self.n * self.n + 1)
        solved[-1] = 0
        solved_plate = solved.reshape((self.n, self.n))
        
        start_h = sum(self._get_single_manhattan(plate[r,c], r, c) for r in range(self.n) for c in range(self.n) if (r,c) not in locked_cells and plate[r,c] != 0)
        start_tuple = tuple(plate.flatten())
        
        open_set = [(start_h, 0, start_h, start_tuple, [])]
        visited = {start_tuple: 0}
        moves = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

        while open_set:
            _, g, h, current_tuple, path = heapq.heappop(open_set)
            curr_plate = np.array(current_tuple).reshape((self.n, self.n))

            if np.array_equal(curr_plate, solved_plate): return path
            r0, c0 = np.argwhere(curr_plate == 0)[0]

            for action, (dr, dc) in moves.items():
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < self.n and 0 <= nc < self.n:
                    if (nr, nc) in locked_cells: continue
                    
                    next_plate = curr_plate.copy()
                    moving_tile = next_plate[nr, nc]
                    next_plate[r0, c0], next_plate[nr, nc] = next_plate[nr, nc], next_plate[r0, c0]
                    
                    if any(next_plate[lr, lc] != solved_plate[lr, lc] for (lr, lc) in locked_cells): continue
                    
                    next_tuple = tuple(next_plate.flatten())
                    new_g = g + 1

                    if next_tuple not in visited or new_g < visited[next_tuple]:
                        visited[next_tuple] = new_g
                        # O(1) Update !
                        new_h = h - self._get_single_manhattan(moving_tile, nr, nc) + self._get_single_manhattan(moving_tile, r0, c0)
                        heapq.heappush(open_set, (new_g + new_h, new_g, new_h, next_tuple, path + [action]))
        return []

    def plan_reduction_strategy(self, start_plate):
        self.action_queue = []
        current_plate = start_plate.copy()
        locked_cells = set()
        for r in range(self.n - 2):
            for c in range(self.n):
                path = self._solve_subproblem(current_plate, locked_cells)
                self.action_queue.extend(path)
                for act in path:
                    r0, c0 = np.argwhere(current_plate == 0)[0]
                    dr, dc = {'U':(-1,0), 'D':(1,0), 'L':(0,-1), 'R':(0,1)}[act]
                    current_plate[r0, c0], current_plate[r0+dr, c0+dc] = current_plate[r0+dr, c0+dc], current_plate[r0, c0]
                locked_cells.add((r, c))
        self.action_queue.extend(self._solve_subproblem(current_plate, locked_cells))

    def choose_action(self, env):
        if not self.action_queue: self.plan_reduction_strategy(env.plate)
        return self.action_queue.pop(0) if self.action_queue else None


class AStarAI(BaseSearchAI):
    def __init__(self, n=3):
        super().__init__(n)
        self.solution_path = []

    def plan_path(self, start_plate):
        solved = np.arange(1, self.n * self.n + 1)
        solved[-1] = 0
        target_tuple = tuple(solved)
        start_tuple = tuple(start_plate.flatten())
        
        start_h = sum(self._get_single_manhattan(start_plate[r,c], r, c) for r in range(self.n) for c in range(self.n) if start_plate[r,c] != 0)
        
        open_set = [(start_h, 0, start_h, start_tuple, [])]
        visited = {start_tuple: 0}
        moves = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

        while open_set:
            _, g, h, current_tuple, path = heapq.heappop(open_set)
            if current_tuple == target_tuple:
                self.solution_path = path
                return
            
            current_plate = np.array(current_tuple).reshape((self.n, self.n))
            r0, c0 = np.argwhere(current_plate == 0)[0]

            for action, (dr, dc) in moves.items():
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < self.n and 0 <= nc < self.n:
                    next_plate = current_plate.copy()
                    moving_tile = next_plate[nr, nc]
                    next_plate[r0, c0], next_plate[nr, nc] = next_plate[nr, nc], next_plate[r0, c0]
                    next_tuple = tuple(next_plate.flatten())
                    
                    new_g = g + 1
                    if next_tuple not in visited or new_g < visited[next_tuple]:
                        visited[next_tuple] = new_g
                        # O(1) Update
                        new_h = h - self._get_single_manhattan(moving_tile, nr, nc) + self._get_single_manhattan(moving_tile, r0, c0)
                        heapq.heappush(open_set, (new_g + new_h, new_g, new_h, next_tuple, path + [action]))

    def choose_action(self, env):
        if not self.solution_path: self.plan_path(env.plate)
        return self.solution_path.pop(0) if self.solution_path else None