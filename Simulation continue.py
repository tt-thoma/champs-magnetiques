import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd
from datetime import datetime
from math import sqrt
import random
import time


class Corps:
    def __init__(self, x, y, vx, vy, ax, ay, m, n, id):
        self.id = id  # Ajouter un identifiant unique
        self.position = [x, y]
        self.vitesse = [vx, vy]
        self.accélération = [ax, ay]
        self.masse = m
        self.P = np.zeros([n, 2])
        self.V = np.zeros([n, 2])
        self.A = np.zeros([n, 2])

    def déplacement(self, dt):
        x = self.position[0]
        y = self.position[1]
        vx = self.vitesse[0]
        vy = self.vitesse[1]
        ax = self.accélération[0]
        ay = self.accélération[1]
        k1_vx = dt * ax
        k1_vy = dt * ay
        k1_x = dt * vx
        k1_y = dt * vy
        k2_vx = dt * (ax + 0.5 * k1_vx)
        k2_vy = dt * (ay + 0.5 * k1_vy)
        k2_x = dt * (vx + 0.5 * k1_vx)
        k2_y = dt * (vy + 0.5 * k1_vy)
        k3_vx = dt * (ax + 0.5 * k2_vx)
        k3_vy = dt * (ay + 0.5 * k2_vy)
        k3_x = dt * (vx + 0.5 * k2_vx)
        k3_y = dt * (vy + 0.5 * k2_vy)
        k4_vx = dt * (x + k3_vx)
        k4_vy = dt * (y + k3_vy)
        k4_x = dt * (vx + k3_vx)
        k4_y = dt * (vy + k3_vy)
        vx = vx + (k1_vx + 2 * k2_vx + 2 * k3_vx + k4_vx) / 6
        vy = vy + (k1_vy + 2 * k2_vy + 2 * k3_vy + k4_vy) / 6
        x = x + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        y = y + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
        self.position[0] = x
        self.position[1] = y
        self.vitesse[0] = vx
        self.vitesse[1] = vy
        self.accélération[0] = ax
        self.accélération[1] = ay

    def données(self, i):
        self.P[i, :] = self.position
        self.V[i, :] = self.vitesse
        self.A[i, :] = self.accélération


class Monde:
    def __init__(self, largeur, hauteur, taille_cellule, temps_simulation):
        self.largeur = largeur
        self.hauteur = hauteur
        self.taille_cellule = taille_cellule
        self.temps_simulation = temps_simulation
        self.plan = np.zeros((largeur // taille_cellule, hauteur // taille_cellule))
        self.corps = []
        self.data = []

    def ajouter_corps(self, corps):
        self.corps.append(corps)

    def simuler(self):
        for t in range(self.temps_simulation):
            for corps in self.corps:
                corps.déplacement(
                    1 / 1000
                )  # Mettre à jour la position, la vitesse et l'accélération
                self.data.append(
                    {
                        "temps": t,
                        "x": corps.position[0],
                        "y": corps.position[1],
                        "vx": corps.vitesse[0],
                        "vy": corps.vitesse[1],
                        "ax": corps.accélération[0],
                        "ay": corps.accélération[1],
                        "corps": corps.id,  # Utiliser l'identifiant pour le groupement
                    }
                )

    def enregistrer_donnees(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        fichier = f"donnees_{timestamp}.csv"
        df = pd.DataFrame(self.data)
        df.to_csv(fichier, index=False)

    def enregistrer_graphiques(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        fichier_vitesse = f"vitesse_{timestamp}.png"
        fichier_acceleration = f"acceleration_{timestamp}.png"
        df = pd.DataFrame(self.data)
        df["vitesse"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)
        df.groupby("corps").vitesse.mean().plot(kind="bar")
        plt.savefig(fichier_vitesse)
        plt.clf()
        df["acceleration"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2)
        df.groupby("corps").acceleration.mean().plot(kind="bar")
        plt.savefig(fichier_acceleration)

    def animer(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        fichier_animation = f"animation_{timestamp}.mp4"
        fig, ax = plt.subplots()
        x = [data["x"] for data in self.data if data["temps"] == 0]
        y = [data["y"] for data in self.data if data["temps"] == 0]
        scat = ax.scatter(x, y)

        temps_debut = time.time()

        def animate(i):
            x = [data["x"] for data in self.data if data["temps"] == i]
            y = [data["y"] for data in self.data if data["temps"] == i]
            scat.set_offsets(np.c_[x, y])

            # Calculer le temps écoulé et estimer le temps restant
            temps_ecoule = time.time() - temps_debut
            temps_restant = (temps_ecoule / (i + 1)) * (self.temps_simulation - i - 1)
            print(f"Il reste environ {temps_restant} secondes.")

        ani = FuncAnimation(fig, animate, frames=self.temps_simulation, repeat=False)
        ani.save(fichier_animation, writer=FFMpegWriter())


monde = Monde(100, 100, 1, 60)
for i in range(10):
    x = random.uniform(-10, 10)
    y = random.uniform(-10, 10)
    vx = random.uniform(-10, 10)
    vy = random.uniform(-10, 10)
    ax = random.uniform(-10, 10)
    ay = random.uniform(-10, 10)
    m = random.uniform(1, 10)
    n = 10000
    c = Corps(x, y, vx, vy, ax, ay, m, n, i)
    monde.ajouter_corps(c)

monde.simuler()
monde.enregistrer_donnees()
monde.enregistrer_graphiques()
monde.animer()
