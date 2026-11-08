#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 22:04:44 2026

@author: codeenlocks
"""

"""
Benchmark Comparatif Taquin 4x4 - Humain vs 2 IA Intelligentes (gui3.py)
Affiche uniquement 3 plateaux côte à côte : Humain, MyStrategyAI et AStarAI.
"""

import sys
import tkinter as tk
import numpy as np

from taquin_env import TaquinEnv
from ai_models import MyStrategyAI, AStarAI

# =========================================================================
# CONFIGURATION DESIGN & COULEURS (Inspiré de gui.py)
# =========================================================================
BG_WOOD = "#C3A378"         # Fond bois clair
TILE_COLOR = "#BA9668"      # Couleur des tuiles (bois café au lait)
TILE_BORDER = "#2B2219"     # Bordure sombre des tuiles
TEXT_COLOR = "#0D0D0D"      # Chiffres noirs très sombres
BOARD_BORDER = "#14100C"    # Bordure extérieure noire du plateau
BG_DARK = "#1E1E1E"        # Fond de la fenêtre Tkinter

DELAY_MS = 400              # Vitesse d'animation automatique (ms)


class MultiTaquinGUI4x4:
    def __init__(self, n=4, path_plate=0):
        self.n = n
        self.path_plate = path_plate
        
        # Fenêtre Tkinter principale
        self.root = tk.Tk()
        self.root.title(f"Taquin {self.n}x{self.n} - Humain vs 2 IA Intelligentes")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        # Taille des cellules adaptée à 3 plateaux côte à côte
        self.cell_size = 110
        self.padding = 6
        self.board_width = self.n * self.cell_size + (self.n + 1) * self.padding
        self.board_height = self.board_width

        # 3 Environnements 4x4 synchronisés
        self.envs = {
            "Humain": TaquinEnv(n=self.n, path=self.path_plate),
            "MyStrategy (Réduction)": TaquinEnv(n=self.n, path=self.path_plate),
            "A* Search (Optimal)": TaquinEnv(n=self.n, path=self.path_plate)
        }

        # Compteurs de coups et états
        self.step_counts = {k: 0 for k in self.envs}
        self.dones = {k: False for k in self.envs}

        # Agents IA (Algébrique + Heuristique)
        self.strategy_agent = MyStrategyAI(n=self.n)
        self.astar_agent = AStarAI(n=self.n)

        # Structures GUI
        self.canvases = {}
        self.status_labels = {}
        self.is_auto_running = False

        self._setup_ui()

        # Binding clavier pour l'humain
        self.root.bind("<Key>", self._on_key_press)

        # Initialisation synchronisée
        self.reset_all()

    def draw_rounded_rect(self, canvas, x1, y1, x2, y2, radius=100, **kwargs):
        """ Dessine un rectangle à coins arrondis sur le canvas. """
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
        # En-tête
        header_lbl = tk.Label(
            self.root,
            text=f"BENCHMARK TAQUIN {self.n}x{self.n} - HUMAIN VS IA",
            font=("Helvetica", 16, "bold"),
            bg=BG_DARK,
            fg="#FFFFFF",
            pady=10
        )
        header_lbl.pack()

        # Conteneur aligné horizontalement (1 ligne de 3 colonnes)
        row_frame = tk.Frame(self.root, bg=BG_DARK)
        row_frame.pack(padx=10, pady=5)

        positions = [
            ("Humain", 0),
            ("MyStrategy (Réduction)", 1),
            ("A* Search (Optimal)", 2)
        ]

        for name, col in positions:
            box = tk.LabelFrame(
                row_frame, 
                text=f"  {name}  ", 
                font=("Helvetica", 11, "bold"),
                bg=BG_DARK,
                fg="#BA9668",
                bd=2,
                relief="groove"
            )
            box.grid(row=0, column=col, padx=10, pady=8)

            lbl = tk.Label(
                box, 
                text="Coups: 0 | En attente", 
                font=("Helvetica", 9), 
                bg=BG_DARK, 
                fg="#CCCCCC"
            )
            lbl.pack(pady=2)
            self.status_labels[name] = lbl

            canvas = tk.Canvas(
                box, 
                width=self.board_width, 
                height=self.board_height, 
                bg=BG_DARK, 
                highlightthickness=0
            )
            canvas.pack(padx=6, pady=6)
            self.canvases[name] = canvas

        # Boutons de contrôle
        btn_frame = tk.Frame(self.root, bg=BG_DARK)
        btn_frame.pack(pady=15)

        self.btn_reset = tk.Button(
            btn_frame, 
            text="🔄 Réinitialiser Plateau 4x4", 
            font=("Helvetica", 10, "bold"),
            bg="#34495E", fg="white", activebackground="#2C3E50",
            command=self.reset_all
        )
        self.btn_reset.pack(side=tk.LEFT, padx=8)

        self.btn_step = tk.Button(
            btn_frame, 
            text="▶️ Étape IA (+1)", 
            font=("Helvetica", 10, "bold"),
            bg="#2980B9", fg="white", activebackground="#3498DB",
            command=self.step_ais
        )
        self.btn_step.pack(side=tk.LEFT, padx=8)

        self.btn_auto = tk.Button(
            btn_frame, 
            text="⚡ Lancer Démo IA", 
            font=("Helvetica", 10, "bold"),
            bg="#BA9668", fg="#14100C", activebackground="#C3A378",
            command=self.toggle_auto
        )
        self.btn_auto.pack(side=tk.LEFT, padx=8)

    def draw_board(self, name):
        """ Redessine le plateau avec le style bois de gui.py. """
        canvas = self.canvases[name]
        canvas.delete("all")
        plate = self.envs[name].plate

        # Plateau extérieur en bois
        self.draw_rounded_rect(
            canvas, 0, 0, self.board_width, self.board_height,
            radius=14, fill=BG_WOOD, outline=BOARD_BORDER, width=3
        )

        # Dessin des tuiles
        for r in range(self.n):
            for c in range(self.n):
                val = plate[r, c]
                x1 = self.padding + c * (self.cell_size + self.padding)
                y1 = self.padding + r * (self.cell_size + self.padding)
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                if val != 0:
                    # Ombre
                    self.draw_rounded_rect(
                        canvas, x1 + 2, y1 + 2, x2 + 2, y2 + 2,
                        radius=8, fill="#1A140E", outline=""
                    )
                    # Tuile
                    self.draw_rounded_rect(
                        canvas, x1, y1, x2, y2,
                        radius=8, fill=TILE_COLOR, outline=TILE_BORDER, width=2
                    )
                    # Texte du numéro
                    font_size = int(self.cell_size * 0.38)
                    canvas.create_text(
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        text=str(val), font=("Helvetica", font_size, "bold"),
                        fill=TEXT_COLOR
                    )

    def reset_all(self):
        """ Recommence une partie synchronisée. """
        self.is_auto_running = False
        self.btn_auto.config(text="⚡ Lancer Démo IA")

        # Chargement d'un plateau 4x4 initial commun
        base_env = TaquinEnv(n=self.n, path=self.path_plate)
        initial_plate = base_env.plate.copy()

        for name in self.envs:
            self.envs[name].plate = initial_plate.copy()
            self.envs[name].current_manhattan = self.envs[name]._get_total_manhattan()
            self.step_counts[name] = 0
            self.dones[name] = False
            self.status_labels[name].config(text="Coups: 0 | En cours...", fg="#CCCCCC")
            self.draw_board(name)

        # Replanification des IA
        self.astar_agent.solution_path = []
        self.astar_agent.plan_path(initial_plate)

        self.strategy_agent.action_queue = []
        self.strategy_agent.plan_reduction_strategy(initial_plate)

    def _on_key_press(self, event):
        """ Déplacements clavier pour l'humain. """
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
        """ Exécute un coup pour MyStrategy et A*. """
        # 1. MyStrategyAI (Réduction Algébrique)
        if not self.dones["MyStrategy (Réduction)"]:
            env = self.envs["MyStrategy (Réduction)"]
            action = self.strategy_agent.choose_action(env)
            if action:
                _, _, done = env.step(action)
                self.step_counts["MyStrategy (Réduction)"] += 1
                self.dones["MyStrategy (Réduction)"] = done
                status = "✅ Résolu !" if done else "En cours..."
                color = "#4CAF50" if done else "#CCCCCC"
                self.status_labels["MyStrategy (Réduction)"].config(
                    text=f"Coups: {self.step_counts['MyStrategy (Réduction)']} | {status}", fg=color
                )
                self.draw_board("MyStrategy (Réduction)")

        # 2. A* Search
        if not self.dones["A* Search (Optimal)"]:
            env = self.envs["A* Search (Optimal)"]
            action = self.astar_agent.choose_action(env)
            if action:
                _, _, done = env.step(action)
                self.step_counts["A* Search (Optimal)"] += 1
                self.dones["A* Search (Optimal)"] = done
                status = "✅ Résolu !" if done else "En cours..."
                color = "#4CAF50" if done else "#CCCCCC"
                self.status_labels["A* Search (Optimal)"].config(
                    text=f"Coups: {self.step_counts['A* Search (Optimal)']} | {status}", fg=color
                )
                self.draw_board("A* Search (Optimal)")

    def toggle_auto(self):
        """ Active ou met en pause l'animation automatique des IA. """
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
            self.dones["MyStrategy (Réduction)"],
            self.dones["A* Search (Optimal)"]
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
    app = MultiTaquinGUI4x4(n=4, path_plate=0)
    app.run()