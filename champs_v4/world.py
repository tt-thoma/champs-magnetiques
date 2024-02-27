import numpy as np
from corps import Particle
from math import sqrt
import constants as const
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy import integrate

class World:
    def __init__(self, size, cell_size , dt: float) -> None:
        self.size: int = size
        self.dt: float = dt
        self.cell_size = cell_size
        self.field_E = np.zeros((size//cell_size, size//cell_size, 2), float)
        self.field_B = np.zeros((size//cell_size, size//cell_size, 2), float)
        
        self.particles: list[Particle] = []

    def add_part(self, part: Particle) -> None:
        self.particles.append(part)
    def calc_E(self):
        # Obtenir la forme du champ électrique
        shape = self.field_E.shape
        
        # Créer une grille de coordonnées pour chaque point dans le monde
        x_coords, y_coords = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        
        # Convertir les coordonnées de la grille en unités de monde
        x_coords = x_coords * self.cell_size
        y_coords = y_coords * self.cell_size
        
        # Extraire les positions x et y ainsi que les charges de toutes les particules
        part_x = np.array([part.x for part in self.particles])
        part_y = np.array([part.y for part in self.particles])
        part_charge = np.array([part.charge for part in self.particles])
        
        # Calculer le vecteur entre chaque point de la grille et chaque particule
        vec_x = x_coords[:, :, None] - part_x
        vec_y = y_coords[:, :, None] - part_y

        # Calculer la norme au carré de ce vecteur (c'est-à-dire sa longueur au carré)
        vec_norm2 = vec_x**2 + vec_y**2 + 1e-9

        # Calculer la norme de ce vecteur (c'est-à-dire sa longueur)
        vec_norm = np.sqrt(vec_norm2)

        # Calculer la norme de la force électrique à chaque point de la grille en utilisant la loi de Coulomb
        force_norm = abs(const.k*part_charge)/vec_norm2

        # Normaliser le vecteur de force pour obtenir un vecteur unitaire dans la direction de la force
        uni_multi = force_norm / vec_norm

        # Convertir vec_x et vec_y en float avant la multiplication
        vec_x = vec_x.astype(float)
        vec_y = vec_y.astype(float)

        # Multiplier le vecteur unitaire par la norme de la force et le signe de la charge
        vec_x *= uni_multi * np.sign(part_charge)
        vec_y *= uni_multi * np.sign(part_charge)

        # Calculer la somme le long de l'axe 2 après avoir empilé vec_x et vec_y
        sum_vec = np.sum(np.dstack((vec_x, vec_y)), axis=2)

        # Redimensionner sum_vec pour qu'il ait la même forme que self.field_E
        sum_vec = np.dstack([sum_vec, sum_vec])

        # Ajouter sum_vec à self.field_E
        self.field_E += sum_vec
        
    def calc_B(self):
        E = self.field_E
        curl_E = np.array(np.gradient(E))[np.array([1,2,0])]-np.array(np.gradient(E))[np.array([2,0,1])]
        dBdt = -curl_E
        B = integrate.simps(dBdt, dx=self.dt)
        self.field_B = B
    
    def E_norm(self):
        # Calculer la norme du champ électrique
        E_norm = np.sqrt(np.sum(self.field_E**2, axis=2))
    
        return E_norm
    def visualize_E(self):
        # Créer une grille de coordonnées
        x = np.arange(0, self.size, self.cell_size)
        y = np.arange(0, self.size, self.cell_size)
        X, Y = np.meshgrid(x, y)

        # Calculer les composantes du champ électrique
        E_x = self.field_E[:,:,0]
        E_y = self.field_E[:,:,1]

        # Créer un diagramme de champ vectoriel
        plt.quiver(X, Y, E_x, E_y)

        # Afficher le diagramme
        plt.show()

w = World(100,1,1)
w.add_part(Particle(50,50,1000000000,0,0))
w.calc_E()
e = w.E_norm()
w.visualize_E()