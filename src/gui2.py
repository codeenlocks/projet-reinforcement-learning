#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 21:46:14 2026

@author: codeenlocks
"""

"""
Interface graphique comparative 2x2 pour le jeu de Taquin (Humain vs 3 IA)
Inspirée du rendu visuel de gui.py (Design bois)
"""

import sys
import tkinter as tk
from tkinter import ttk
import numpy as np

from taquin_env import TaquinEnv
from ai_models import QLearningAI, MyStrategyAI, AStarAI

# =========================================================================
# CONFIGURATION DESIGN & COULEURS (Alignées sur gui.py)
# =========================================================================
BG_WOOD = "#C3A378"         # Fond bois clair
TILE_COLOR = "#BA9668"      # Couleur des tuiles (bois café au lait)
TILE_BORDER = "#2B2219"     # Bordure sombre des tuiles
TEXT_COLOR = "#0D0D0D"      # Chiffres noirs très sombres
BOARD_BORDER = "#14100C"    # Bordure extérieure noire du plateau
BG_DARK = "#1E1E1E"        # Fond de la fenêtre Tkinter

DELAY_MS = 500              # Vitesse d'animation automatique (ms)


class MultiTaquinGUI:
    def __init__(self, n=3, version=2, path_plate=0):
        self.n = n
        self.path_plate = path_plate
        
        # Initialisation de la fenêtre Tkinter principale
        self.root = tk.Tk()
        self.root.title(f"Taquin {self.n}x{self.n} - Benchmark Comparatif (Humain vs 3 IA)")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        # Taille des cellules adaptée à l'affichage 2x2 (4 plateaux)
        self.cell_size = 65 if self.n >= 3 else 90
        self.padding = 8
        self.board_width = self.n * self.cell_size + (self.n + 1) * self.padding
        self.board_height = self.board_width

        # Environnements synchronisés
        self.envs = {
            "Humain": TaquinEnv(n=self.n, path=self.path_plate),
            "Q-Learning": TaquinEnv(n=self.n, path=self.path_plate),
            "MyStrategy": TaquinEnv(n=self.n, path=self.path_plate),
            "A* Search": TaquinEnv(n=self.n, path=self.path_plate)
        }

        # Compteurs de coups et états
        self.step_counts = {k: 0 for k in self.envs}
        self.dones = {k: False for k in self.envs}

        # Initialisation des agents IA
        self.q_agent = QLearningAI(n=self.n, exploration_rate=0.0)
        # Tentative de chargement (PKL en priorité, fallback JSON)
        if not self.q_agent.load_q_table(version=version):
            print(f"[Avertissement] Fichier PKL v{version} introuvable, essai du JSON...")
            self.q_agent.load_q_table_json(version=version)

        self.strategy_agent = MyStrategyAI(n=self.n)
        self.astar_agent = AStarAI(n=self.n)

        # Structures GUI
        self.canvases = {}
        self.status_labels = {}
        self.is_auto_running = False

        self._setup_ui()

        # Binding clavier pour le joueur humain
        self.root.bind("<Key>", self._on_key_press)

        # Démarrage avec un plateau initial identique pour tous
        self.reset_all()

    def draw_rounded_rect(self, canvas, x1, y1, x2, y2, radius=12, **kwargs):
        """ Dessine un rectangle à coins arrondis sur un canvas donné. """
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1
        ]
        return canvas.create_polygon(points, **kwargs, smooth=True)

    def _setup_ui(self):
        # En-tête principal
        header_lbl = tk.Label(
            self.root,
            text=f"BENCHMARK TAQUIN {self.n}x{self.n}",
            font=("Helvetica", 16, "bold"),
            bg=BG_DARK,
            fg="#FFFFFF",
            pady=10
        )
        header_lbl.pack()

        # Cadre conteneur pour la grille 2x2
        grid_frame = tk.Frame(self.root, bg=BG_DARK)
        grid_frame.pack(padx=15, pady=5)

        positions = [
            ("Humain", 0, 0),
            ("Q-Learning", 0, 1),
            ("MyStrategy", 1, 0),
            ("A* Search", 1, 1)
        ]

        for name, row, col in positions:
            box = tk.LabelFrame(
                grid_frame, 
                text=f"  {name}  ", 
                font=("Helvetica", 11, "bold"),
                bg=BG_DARK,
                fg="#BA9668",
                bd=2,
                relief="groove"
            )
            box.grid(row=row, column=col, padx=10, pady=10)

            # Label statut/coups sous le titre
            lbl = tk.Label(
                box, 
                text="Coups: 0 | En attente", 
                font=("Helvetica", 10), 
                bg=BG_DARK, 
                fg="#CCCCCC"
            )
            lbl.pack(pady=2)
            self.status_labels[name] = lbl

            # Canvas du plateau
            canvas = tk.Canvas(
                box, 
                width=self.board_width, 
                height=self.board_height, 
                bg=BG_DARK, 
                highlightthickness=0
            )
            canvas.pack(padx=8, pady=8)
            self.canvases[name] = canvas

        # Zone de boutons de contrôle en bas
        btn_frame = tk.Frame(self.root, bg=BG_DARK)
        btn_frame.pack(pady=15)

        self.btn_reset = tk.Button(
            btn_frame, 
            text="🔄 Réinitialiser Plateau", 
            font=("Helvetica", 11, "bold"),
            bg="#34495E", fg="white", activebackground="#2C3E50",
            command=self.reset_all
        )
        self.btn_reset.pack(side=tk.LEFT, padx=10)

        self.btn_step = tk.Button(
            btn_frame, 
            text="▶️ Étape IA (+1)", 
            font=("Helvetica", 11, "bold"),
            bg="#2980B9", fg="white", activebackground="#3498DB",
            command=self.step_ais
        )
        self.btn_step.pack(side=tk.LEFT, padx=10)

        self.btn_auto = tk.Button(
            btn_frame, 
            text="⚡ Lancer Démo IA", 
            font=("Helvetica", 11, "bold"),
            bg="#BA9668", fg="#14100C", activebackground="#C3A378",
            command=self.toggle_auto
        )
        self.btn_auto.pack(side=tk.LEFT, padx=10)

    def draw_board(self, name):
        """ Redessine le plateau d'une instance spécifique avec le style bois gui.py. """
        canvas = self.canvases[name]
        canvas.delete("all")
        plate = self.envs[name].plate

        # 1. Fond du plateau en bois
        self.draw_rounded_rect(
            canvas, 0, 0, self.board_width, self.board_height,
            radius=18, fill=BG_WOOD, outline=BOARD_BORDER, width=4
        )

        # 2. Dessin des tuiles
        for r in range(self.n):
            for c in range(self.n):
                val = plate[r, c]
                x1 = self.padding + c * (self.cell_size + self.padding)
                y1 = self.padding + r * (self.cell_size + self.padding)
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                if val != 0:
                    # Ombre portée
                    self.draw_rounded_rect(
                        canvas, x1 + 2, y1 + 3, x2 + 2, y2 + 3,
                        radius=10, fill="#1A140E", outline=""
                    )
                    # Tuile
                    self.draw_rounded_rect(
                        canvas, x1, y1, x2, y2,
                        radius=10, fill=TILE_COLOR, outline=TILE_BORDER, width=2
                    )
                    # Chiffre
                    font_size = int(self.cell_size * 0.38)
                    canvas.create_text(
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        text=str(val),
                        font=("Helvetica", font_size, "bold"),
                        fill=TEXT_COLOR
                    )

    def reset_all(self):
        """ Recommence une partie sur un plateau synchronisé. """
        self.is_auto_running = False
        self.btn_auto.config(text="⚡ Lancer Démo IA")

        # Chargement/Génération d'un plateau initial commun
        base_env = TaquinEnv(n=self.n, path=self.path_plate)
        initial_plate = base_env.plate.copy()

        for name in self.envs:
            self.envs[name].plate = initial_plate.copy()
            self.envs[name].current_manhattan = self.envs[name]._get_total_manhattan()
            self.step_counts[name] = 0
            self.dones[name] = False
            self.status_labels[name].config(text="Coups: 0 | En cours...", fg="#CCCCCC")
            self.draw_board(name)

        # Replanification des parcours IA
        self.astar_agent.solution_path = []
        self.astar_agent.plan_path(initial_plate)

        self.strategy_agent.action_queue = []
        self.strategy_agent.plan_reduction_strategy(initial_plate)

    def _on_key_press(self, event):
        """ Gestion des touches de direction pour le joueur humain. """
        if self.dones["Humain"]:
            return

        key_map = {"Up": "U", "Down": "D", "Left": "L", "Right": "R"}
        if event.keysym in key_map:
            action = key_map[event.keysym]
            valid = self.envs["Humain"].get_valid_actions()
            if action in valid:
                _, _, done = self.envs["Humain"].step(action)
                self.step_counts["Humain"] += 1
                self.dones["Humain"] = done

                status = "✅ Résolu !" if done else "En cours..."
                color = "#4CAF50" if done else "#CCCCCC"
                self.status_labels["Humain"].config(
                    text=f"Coups: {self.step_counts['Humain']} | {status}", fg=color
                )
                self.draw_board("Humain")

    def step_ais(self):
        """ Effectue exactement un coup pour chacune des 3 IA. """
        # 1. Q-Learning
        if not self.dones["Q-Learning"]:
            env = self.envs["Q-Learning"]
            # Correct : plus d'argument 'eps' passé ici
            action = self.q_agent.choose_action(env.plate, env.get_valid_actions())
            _, _, done = env.step(action)
            self.step_counts["Q-Learning"] += 1
            self.dones["Q-Learning"] = done
            status = "✅ Résolu !" if done else "En cours..."
            color = "#4CAF50" if done else "#CCCCCC"
            self.status_labels["Q-Learning"].config(
                text=f"Coups: {self.step_counts['Q-Learning']} | {status}", fg=color
            )
            self.draw_board("Q-Learning")

        # 2. MyStrategyAI (Algébrique / Réduction)
        if not self.dones["MyStrategy"]:
            env = self.envs["MyStrategy"]
            action = self.strategy_agent.choose_action(env)
            if action:
                _, _, done = env.step(action)
                self.step_counts["MyStrategy"] += 1
                self.dones["MyStrategy"] = done
                status = "✅ Résolu !" if done else "En cours..."
                color = "#4CAF50" if done else "#CCCCCC"
                self.status_labels["MyStrategy"].config(
                    text=f"Coups: {self.step_counts['MyStrategy']} | {status}", fg=color
                )
                self.draw_board("MyStrategy")

        # 3. A* Search
        if not self.dones["A* Search"]:
            env = self.envs["A* Search"]
            action = self.astar_agent.choose_action(env)
            if action:
                _, _, done = env.step(action)
                self.step_counts["A* Search"] += 1
                self.dones["A* Search"] = done
                status = "✅ Résolu !" if done else "En cours..."
                color = "#4CAF50" if done else "#CCCCCC"
                self.status_labels["A* Search"].config(
                    text=f"Coups: {self.step_counts['A* Search']} | {status}", fg=color
                )
                self.draw_board("A* Search")

    def toggle_auto(self):
        """ Démarre ou arrête la résolution automatique pas à pas. """
        self.is_auto_running = not self.is_auto_running
        if self.is_auto_running:
            self.btn_auto.config(text="⏸️ Pause")
            self._auto_loop()
        else:
            self.btn_auto.config(text="⚡ Lancer Démo IA")

    def _auto_loop(self):
        if not self.is_auto_running:
            return

        all_dones = all([
            self.dones["Q-Learning"],
            self.dones["MyStrategy"],
            self.dones["A* Search"]
        ])

        if not all_dones:
            self.step_ais()
            self.root.after(DELAY_MS, self._auto_loop)
        else:
            self.is_auto_running = False
            self.btn_auto.config(text="⚡ Lancer Démo IA")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    # n=3 (Taquin 3x3), version=3 (Charge q_table_v3.pkl/json), path_plate=0
    app = MultiTaquinGUI(n=3, version='final', path_plate=0)
    app.run()