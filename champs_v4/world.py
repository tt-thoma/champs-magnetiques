import numpy as np
from corps import Particle
from math import sqrt
import constants as const
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import datetime
import matplotlib.animation as animation
import os
import tqdm
import sys
import os
import shutil

class World:
    def __init__(self, size, cell_size, dt: float) -> None:
        self.size: int = size
        self.dt: float = dt
        self.cell_size = cell_size

        size_int = int(size // cell_size)
        self.field_E = np.zeros((size_int, size_int, size_int, 3), dtype=float)
        self.field_B = np.zeros((size_int, size_int, size_int, 3), dtype=float)

        self.particles: list[Particle] = []
        self.temps = 0

    def add_part(self, part: Particle) -> None:
        self.particles.append(part)

    def calc_E(self):
        # Début du calcul du champ électrique engendré par chaque particule...
        shape = self.field_E.shape

        # Créer une grille de coordonnées pour chaque point dans le monde
        x_coords, y_coords, z_coords = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
        )

        # Convertir les coordonnées de la grille en unités de monde
        x_coords = x_coords * self.cell_size
        y_coords = y_coords * self.cell_size
        z_coords = z_coords * self.cell_size

        # Initialiser le champ électrique total
        total_E = np.zeros_like(self.field_E)

        # Parcourir chaque particule
        for part in self.particles:
            # Calculer le vecteur de position entre la particule et chaque point de la grille
            vec_pos = np.stack(
                (x_coords - part.x, y_coords - part.y, z_coords - part.z), axis=-1
            )

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

    def calc_B(self):
        # Début du calcul du champ magnétique...
        dB_dt = np.diff(self.field_E, axis=-1) / self.dt

        self.field_B[..., 0] = -dB_dt[..., 0]

        self.field_B[..., 1:] = -integrate.cumtrapz(
            dB_dt, dx=self.dt, initial=0, axis=-1
        )

    def plot_E_field(self):
        # Début de la représentation du champ électrique sous forme de vecteurs...
        norm_E = np.linalg.norm(self.field_E, axis=-1)
        alpha = 0 + 0.6 * (1 - norm_E / np.max(norm_E))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("black")  

        Ex = self.field_E[..., 0]
        Ey = self.field_E[..., 1]
        Ez = self.field_E[..., 2]

        x_coords, y_coords, z_coords = np.meshgrid(
            np.arange(0, self.size, self.cell_size),
            np.arange(0, self.size, self.cell_size),
            np.arange(0, self.size, self.cell_size),
        )

        colors = norm_E.flatten()

        ax.quiver(
            x_coords.flatten(),
            y_coords.flatten(),
            z_coords.flatten(),
            Ex.flatten(),
            Ey.flatten(),
            Ez.flatten(),
            length=0.5,
            normalize=True,
            alpha=alpha.flatten(),
            color=plt.cm.viridis(colors / np.max(colors)),
        )

        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=np.max(colors))
        )
        sm.set_array(colors)
        cbar = plt.colorbar(sm, ax=ax, pad=0.05)
        cbar.set_label("Norme du champ électrique")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.zaxis.label.set_color("white")

        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.tick_params(axis="z", colors="white")

        plt.show()

    def calc_next(self):
        self.calc_E()
        self.calc_B()
        for part in self.particles:
            part.calc_next(self.field_E, self.field_B, self.size, self.dt)
        self.temps += self.dt

    def plot_B_field(self):
        # Début de la représentation du champ magnétique sous forme de vecteurs...
        norm_B = np.linalg.norm(self.field_B, axis=-1)
        alpha = 0.4 + 0.6 * (1 - norm_B / np.max(norm_B))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("black")  

        Bx = self.field_B[..., 0]
        By = self.field_B[..., 1]
        Bz = self.field_B[..., 2]

        shape = self.field_B.shape[:-1] 
        grid_size = [np.arange(0, s, self.cell_size) for s in shape]
        x_coords, y_coords, z_coords = np.meshgrid(*grid_size, indexing="ij")

        colors = norm_B.flatten()

        ax.quiver(
            x_coords.flatten(),
            y_coords.flatten(),
            z_coords.flatten(),
            Bx.flatten(),
            By.flatten(),
            Bz.flatten(),
            length=0.5,
            normalize=True,
            alpha=alpha.flatten(),
            color=plt.cm.viridis(colors / np.max(colors)),
        )

        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=np.max(colors))
        )
        sm.set_array(colors)
        cbar = plt.colorbar(sm, ax=ax, pad=0.05)
        cbar.set_label("Norme du champ magnétique")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.zaxis.label.set_color("white")

        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.tick_params(axis="z", colors="white")

        plt.show()

    def create_animation(self, fps, total_simulation_time, total_animation_time, output_folder_name):
        current_time = datetime.datetime.now()  
        current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")  
        filename = f"Simulation_{self.size}_{self.cell_size}_{self.dt}_{fps}fps_{current_time_str}.mp4"

        output_folder = os.path.join(output_folder_name, f"Simulation_{self.size}_{self.cell_size}_{self.dt}_{fps}fps_{current_time_str}")

        os.makedirs(output_folder, exist_ok=True)

        animation_output_path = os.path.join(output_folder, filename)

        console_output_path = os.path.join(output_folder, f"Simulation_{self.size}_{self.cell_size}_{self.dt}_{fps}fps_{current_time_str}_console.txt")

        with open(console_output_path, "w") as console_file:
            sys.stdout = console_file

            print(f"Paramètres de simulation : size={self.size}, cell_size={self.cell_size}, dt={self.dt}")
            print(f"FPS de l'animation : {fps}")
            print(f"Durée totale de la simulation : {total_simulation_time} s")
            print(f"Durée totale de l'animation : {total_animation_time} s")

            sys.stdout = sys.__stdout__

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        def update(frame):
            ax.clear()
            ax.set_xlim(0, self.size)
            ax.set_ylim(0, self.size)
            ax.set_zlim(0, self.size)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Simulation - Temps écoulé: {frame / fps} s')

            for part in self.particles:
                ax.scatter(part.x, part.y, part.z, c='r', marker='o')

            self.calc_next()  

        total_animation_frames = int(total_animation_time * fps)  
        fps_interval = 1000 / fps  

        with tqdm.tqdm(total=total_animation_frames, desc="Calcul de l'animation", unit="frames") as progress_bar:
            def update_with_progress(frame):
                update(frame)
                progress_bar.update(1)  

            ani = animation.FuncAnimation(fig, update_with_progress, frames=total_animation_frames, interval=fps_interval)
        
            ani.save(animation_output_path, writer='ffmpeg', fps=fps)

            print(f"L'animation a été enregistrée sous {animation_output_path}")
            print(f"Les informations sont également enregistrées dans {console_output_path}")


# Créer une instance de la classe World
w = World(10,1, 0.001)
w.add_part(Particle(5, 5, 1,-1000000*const.charge_electron, 0,0,0))
w.add_part(Particle(6, 6, 2,1000000*const.charge_electron, 0,0,0))
fps = 30  
duree_simulation = 0.1
duree_animation = 11
clear = True
def simulation(t=duree_simulation):
    global w
    while w.temps < duree_simulation:
        w.calc_next()

        """print(
            f"position de la particule 1 {w.particles[1].x, w.particles[1].y, w.particles[1].z}"""
        
        print(f"Temps écoulé : {w.temps} / {duree_simulation}")

    print("Simulation terminée !")

# Appeler la méthode create_animation avec les arguments spécifiés
w.create_animation(fps, duree_simulation, duree_animation, output_folder_name="Résultats")


def crf():
    folder_path = "Résultats"
    """
    Clear all files and subfolders in the specified folder.

    Args:
        folder_path (str): Path to the folder to be cleared.
    """
    # Vérifie si le chemin est un dossier existant
    if not os.path.isdir(folder_path):
        print(f"Le chemin spécifié '{folder_path}' n'est pas un dossier existant.")
        return

    try:
        # Parcourir tous les éléments (fichiers et sous-dossiers) dans le dossier
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            # Si c'est un dossier, récursivement supprimer son contenu
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            # Si c'est un fichier, le supprimer
            else:
                os.remove(item_path)
        print(f"Le dossier '{folder_path}' a été vidé avec succès.")
    except Exception as e:
        print(f"Erreur lors de la suppression du contenu du dossier '{folder_path}': {e}")
if clear == True :
    crf()
