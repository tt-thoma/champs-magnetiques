import numpy as np
import time

class Univers:
    def __init__(self, taille, facteur_de_discretion) -> None:
        self.taille = taille # en mètre
        self.facteur_de_discretion = facteur_de_discretion
        self.l = self.taille * self.facteur_de_discretion
        self.espace = np.zeros([self.l, self.l, self.l])

    def calc(self):
        d1 = 49 * self.facteur_de_discretion
        for (x, y, z), _ in np.ndenumerate(self.espace):
            x *= self.facteur_de_discretion
            y *= self.facteur_de_discretion
            z *= self.facteur_de_discretion
            denominateur = ((d1-x)**2 + (d1-y)**2 + (d1-z)**2)
            f = 1000 if denominateur == 0 else (9*10**9 * 1.6*10**-19) / denominateur
            x = round(x / self.facteur_de_discretion)
            y = round(y / self.facteur_de_discretion)
            z = round(z / self.facteur_de_discretion)
            self.espace[x, y, z] = f

# Début du chronomètre
start_time = time.time()

u1 = Univers(1, 100)
u1.calc()

# Fin du chronomètre
end_time = time.time()

# Temps d'exécution
execution_time = end_time - start_time

print(f"Le temps d'exécution est {execution_time} secondes.")
