#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:42:13 2026

@author: codeenlocks
"""


import sys
import time
import tkinter as tk
from taquin_env import TaquinEnv
from ai_models import QLearningAI

# =========================================================================
# CONFIGURATION DESIGN & COULEURS (Inspiré du visuel)
# =========================================================================
BG_WOOD = "#C3A378"         # Fond bois clair
TILE_COLOR = "#BA9668"      # Couleur des tuiles (bois café au lait)
TILE_BORDER = "#2B2219"     # Bordure sombre des tuiles
TEXT_COLOR = "#0D0D0D"      # Chiffres noirs très sombres
BOARD_BORDER = "#14100C"    # Bordure extérieure noire du plateau

DELAY_MS = 600              # Vitesse d'animation (millisecondes par coup)


class TaquinGUI:
    def __init__(self, n=2, version=1, path_plate=0):
        self.n = n
        
        # Initialisation de l'environnement
        self.env = TaquinEnv(n=self.n, path=path_plate)
        
        # Initialisation de l'agent en mode pure exploitation (eps = 0.0)
        self.agent = QLearningAI(n=self.n, exploration_rate=0.0)
        success = self.agent.load_q_table(version=version)
        
        if not success:
            print("[ERREUR GUI] Impossible de charger la Q-Table. Abandon.")
            sys.exit(1)

        # Fenêtre principale Tkinter
        self.root = tk.Tk()
        self.root.title(f"Taquin {self.n}x{self.n} - Résolution IA")
        self.root.configure(bg="#1E1E1E")
        self.root.resizable(False, False)

        # Dimensions du Canvas
        self.cell_size = 110
        self.padding = 12
        self.board_width = self.n * self.cell_size + (self.n + 1) * self.padding
        self.board_height = self.board_width

        # Zone d'en-tête (Titre et status)
        self.status_label = tk.Label(
            self.root, 
            text="Initialisation...", 
            font=("Helvetica", 14, "bold"), 
            bg="#1E1E1E", 
            fg="#FFFFFF", 
            pady=10
        )
        self.status_label.pack()

        # Canvas pour le rendu personnalisé du plateau bois
        self.canvas = tk.Canvas(
            self.root, 
            width=self.board_width, 
            height=self.board_height, 
            bg="#1E1E1E", 
            highlightthickness=0
        )
        self.canvas.pack(padx=20, pady=10)

        # Compteur de mouvements
        self.step_count = 0
        self.is_running = False

        # Affichage du plateau initial
        self.draw_board()

        # Bouton pour démarrer la résolution par l'IA
        self.btn_start = tk.Button(
            self.root, 
            text="Lancer l'IA", 
            font=("Helvetica", 12, "bold"), 
            bg="#BA9668", 
            fg="#14100C", 
            activebackground="#C3A378", 
            command=self.start_ai
        )
        self.btn_start.pack(pady=15)

    def draw_rounded_rect(self, x1, y1, x2, y2, radius=15, **kwargs):
        """ Dessine un rectangle à coins arrondis sur le Canvas. """
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
        return self.canvas.create_polygon(points, **kwargs, smooth=True)

    def draw_board(self):
        """ Redessine le plateau et ses tuiles avec le design bois. """
        self.canvas.delete("all")

        # 1. Conteneur extérieur (Plateau Bois)
        self.draw_rounded_rect(
            0, 0, self.board_width, self.board_height, 
            radius=25, fill=BG_WOOD, outline=BOARD_BORDER, width=6
        )

        # 2. Dessin des tuiles
        plate = self.env.plate
        for r in range(self.n):
            for c in range(self.n):
                val = plate[r, c]
                x1 = self.padding + c * (self.cell_size + self.padding)
                y1 = self.padding + r * (self.cell_size + self.padding)
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                if val != 0:
                    # Ombre portée sous la tuile
                    self.draw_rounded_rect(
                        x1 + 3, y1 + 4, x2 + 3, y2 + 4, 
                        radius=16, fill="#1A140E", outline=""
                    )
                    # Tuile principale
                    self.draw_rounded_rect(
                        x1, y1, x2, y2, 
                        radius=16, fill=TILE_COLOR, outline=TILE_BORDER, width=3
                    )
                    # Texte du numéro
                    font_size = int(self.cell_size * 0.38)
                    self.canvas.create_text(
                        (x1 + x2) / 2, (y1 + y2) / 2, 
                        text=str(val), 
                        font=("Helvetica", font_size, "bold"), 
                        fill=TEXT_COLOR
                    )

    def start_ai(self):
        """ Démarre la boucle de résolution automatique. """
        if self.is_running:
            return
        
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.step_count = 0
        
        print("\n==================================================")
        print(f"--- DÉBUT DÉMONSTRATION IA (TAQUIN {self.n}x{self.n}) ---")
        print("==================================================")
        print("[ÉTAPES] Configuration initiale du plateau :")
        self.env.render()
        
        self.status_label.config(text="Résolution en cours par l'IA...")
        self.root.after(500, self.ai_step)

    def ai_step(self):
        """ Exécute un coup calculé par la Q-Table et met à jour le GUI et la console. """
        state = self.env.plate
        valid_actions = self.env.get_valid_actions()
        action = self.agent.choose_action(state, valid_actions)

        # Action dans l'environnement
        next_state, reward, done = self.env.step(action)
        self.step_count += 1

        # Affichage dans la console
        print(f"Coup n°{self.step_count} : Mouvement '{action}'")
        self.env.render()

        # Mise à jour graphique
        self.draw_board()

        if done:
            print("--------------------------------------------------")
            print(f"[SUCCÈS] Taquin résolu en {self.step_count} mouvements !")
            print("--------------------------------------------------\n")
            self.status_label.config(
                text=f"Résolu en {self.step_count} coups !", fg="#4CAF50"
            )
            self.is_running = False
        elif self.step_count >= 100:
            print("--------------------------------------------------")
            print("[ÉCHEC] Nombre maximal de coups atteint (IA bloquée).")
            print("--------------------------------------------------\n")
            self.status_label.config(text="Échec (Limite atteinte)", fg="#F44336")
            self.is_running = False
        else:
            # Prochain coup après le délai
            self.root.after(DELAY_MS, self.ai_step)

    def run(self):
        """ Lance la boucle d'événements GUI. """
        self.root.mainloop()


if __name__ == "__main__":
    # N=2 (Taquin 2x2), version=1 (Q-Table q_table_v1.pkl), path_plate=0 (plateau aléatoire dans plates/2x2/)
    app = TaquinGUI(n=3, version='final', path_plate=0)
    app.run()