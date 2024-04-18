import numpy as np
from corps import Particle
from math import sqrt
import constants as const
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
class World:
    def __init__(self, size, cell_size, dt: float) -> None:
        self.size: int = size
        self.dt: float = dt
        self.cell_size = cell_size
        self.field_E = np.zeros((size // cell_size, size // cell_size, size // cell_size, 3), float)
        self.field_B = np.zeros((size // cell_size, size // cell_size, size // cell_size, 3), float)
        self.particles: list[Particle] = []

       

    def add_part(self, part: Particle) -> None:
        self.particles.append(part)

    def calc_E(self):
        print("Début du calcul du champ électrique engendré par chaque particule...")

        # Obtenir la forme du champ électrique
        shape = self.field_E.shape

        # Créer une grille de coordonnées pour chaque point dans le monde
        x_coords, y_coords, z_coords = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')

        # Convertir les coordonnées de la grille en unités de monde
        x_coords = x_coords * self.cell_size
        y_coords = y_coords * self.cell_size
        z_coords = z_coords * self.cell_size

        # Initialiser le champ électrique total
        total_E = np.zeros_like(self.field_E)

        # Parcourir chaque particule
        for part in self.particles:
            # Calculer le vecteur de position entre la particule et chaque point de la grille
            vec_pos = np.stack((x_coords - part.x, y_coords - part.y, z_coords - part.z), axis=-1)

            # Calculer la distance entre la particule et chaque point de la grille
            distance = np.linalg.norm(vec_pos, axis=-1)

            # Remplacer les valeurs nulles dans distance par une petite valeur non nulle pour éviter la division par zéro
            distance = np.where(distance == 0, 1e-9, distance)

            # Calculer la norme du champ électrique à chaque point de la grille en utilisant la loi de Coulomb
            E_norm = const.k * part.charge / (distance**2)

            # Calculer le vecteur de champ électrique à chaque point de la grille
            E_vec = E_norm[..., np.newaxis] * (vec_pos / distance[..., np.newaxis])

            # Ajouter la contribution de cette particule au champ électrique total
            total_E += E_vec

        # Assigner le champ électrique total calculé à self.field_E
        self.field_E = total_E

        print("Calcul du champ électrique engendré par chaque particule terminé.")






    def calc_B(self):
        print("Début du calcul du champ magnétique...")

        # Calcul de la dérivée spatiale du champ électrique (curl E)
        curl_E = (
            np.array(np.gradient(self.field_E))[np.array([1, 2, 0])]
            - np.array(np.gradient(self.field_E))[np.array([2, 0, 1])]
        )

        # Intégration temporelle de dB/dt pour obtenir B
        B = integrate.simps(curl_E, dx=self.dt, axis=-1)

        # Assigner le champ magnétique calculé à self.field_B
        self.field_B = B

        print("Calcul du champ magnétique terminé.")
    def plot_E_field(self):
        print("Début de la représentation du champ électrique sous forme de vecteurs...")

        # Calcul de la norme du champ électrique
        norm_E = np.linalg.norm(self.field_E, axis=-1)

        # Calculer l'alpha en fonction de la norme, avec un minimum de 40%
        alpha =0 + 0.6 * (1 - norm_E / np.max(norm_E))

        # Créer une figure 3D avec un fond transparent
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor('black')  # Définir le fond du graphique en noir

        # Extraire les composantes x, y et z du champ électrique
        Ex = self.field_E[..., 0]
        Ey = self.field_E[..., 1]
        Ez = self.field_E[..., 2]

        # Créer une grille de coordonnées pour les vecteurs
        x_coords, y_coords, z_coords = np.meshgrid(
            np.arange(0, self.size, self.cell_size),
            np.arange(0, self.size, self.cell_size),
            np.arange(0, self.size, self.cell_size)
        )

        # Calculer la couleur basée sur la norme du champ électrique
        colors = norm_E.flatten()

        # Tracer les vecteurs du champ électrique avec couleur basée sur la norme et alpha ajusté
        ax.quiver(x_coords.flatten(), y_coords.flatten(), z_coords.flatten(), 
                Ex.flatten(), Ey.flatten(), Ez.flatten(), 
                length=0.5, normalize=True, alpha=alpha.flatten(), color=plt.cm.viridis(colors / np.max(colors)))

        # Ajouter une barre de couleur
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=np.max(colors)))
        sm.set_array(colors)
        cbar = plt.colorbar(sm, ax=ax, pad=0.05)
        cbar.set_label('Norme du champ électrique')

        # Étiqueter les axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Configurer la couleur du texte des axes
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')

        # Configurer la couleur des traits des axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')

        # Afficher la figure
        plt.show()

        print("Représentation du champ électrique sous forme de vecteurs terminée.")
    def plot_B_field(self):
        print("Début de la représentation du champ magnétique sous forme de vecteurs...")

        # Calcul de la norme du champ magnétique
        norm_B = np.linalg.norm(self.field_B, axis=-1)

        # Calculer l'alpha en fonction de la norme, avec un minimum de 40%
        alpha = 0.4 + 0.6 * (1 - norm_B / np.max(norm_B))

        # Créer une figure 3D avec un fond transparent
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor('black')  # Définir le fond du graphique en noir

        # Extraire les composantes x, y et z du champ magnétique
        Bx = self.field_B[..., 0]
        By = self.field_B[..., 1]
        Bz = self.field_B[..., 2]

        # Créer une grille de coordonnées pour les vecteurs
        x_coords, y_coords, z_coords = np.meshgrid(
            np.arange(0, self.size, self.cell_size),
            np.arange(0, self.size, self.cell_size),
            np.arange(0, self.size, self.cell_size)
        )

        # Calculer la couleur basée sur la norme du champ magnétique
        colors = norm_B.flatten()

        # Tracer les vecteurs du champ magnétique avec couleur basée sur la norme et alpha ajusté
        ax.quiver(x_coords.flatten(), y_coords.flatten(), z_coords.flatten(), 
                Bx.flatten(), By.flatten(), Bz.flatten(), 
                length=0.5, normalize=True, alpha=alpha.flatten(), color=plt.cm.viridis(colors / np.max(colors)))

        # Ajouter une barre de couleur
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=np.max(colors)))
        sm.set_array(colors)
        cbar = plt.colorbar(sm, ax=ax, pad=0.05)
        cbar.set_label('Norme du champ magnétique')

        # Étiqueter les axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Configurer la couleur du texte des axes
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')

        # Configurer la couleur des traits des axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')

        # Afficher la figure
        plt.show()

        print("Représentation du champ magnétique sous forme de vecteurs terminée.")

        

w = World(10, 1, 1)
w.add_part(Particle(3, 3, 3, 100000000, 0, 0))
w.add_part(Particle(7, 7, 7, -100000000, 0, 0))
print(w.field_E.shape)
w.calc_E()
w.calc_B()
print(w.field_E.shape)
#w.plot_E_field()
w.plot_B_field()
