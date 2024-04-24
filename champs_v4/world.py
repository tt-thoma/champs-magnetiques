import numpy as np
import constants as const
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import tqdm
import sys
import shutil
import datetime
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
import logging

from corps import Particle

class World:
    
    def __init__(self, size, cell_size, dt: float) -> None:
        self.size: int = size
        self.dt: float = dt
        self.cell_size = cell_size

        size_int = int(size // cell_size)
        self.field_E = np.zeros((size_int, size_int, size_int, 3), dtype=np.float64)
        self.field_B = np.zeros((size_int, size_int, size_int, 3), dtype=np.float64)

        self.particles: list[Particle] = []
        self.temps = 0

    def add_part(self, part: Particle) -> None:
        self.particles.append(part)

    def calc_E(self):
        shape = self.field_E.shape
        x_coords, y_coords, z_coords = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
        )

        x_coords = x_coords * self.cell_size
        y_coords = y_coords * self.cell_size
        z_coords = z_coords * self.cell_size

        total_E = np.zeros_like(self.field_E)

        for part in self.particles:
            vec_pos = np.stack(
                (x_coords - part.x, y_coords - part.y, z_coords - part.z), axis=-1
            )
            distance = np.linalg.norm(vec_pos, axis=-1)
            
            # Exclure la contribution si la distance est nulle (particule à la même position)
            valid_distance = distance != 0
            
            # Utiliser np.newaxis pour étendre la dimension sur laquelle nous appliquons le masque booléen
            E_norm = const.k * part.charge / (distance**2).astype(np.float64)
            E_vec = E_norm[..., np.newaxis] * vec_pos / distance[..., np.newaxis]
            
            # Appliquer le masque uniquement sur les dimensions appropriées
            total_E[valid_distance] += E_vec[valid_distance]



        self.field_E = total_E
        if np.isnan(self.field_E).any():
            print(f"Le champ électrique contient des valeurs NaN après le calcul.")
            
    def calc_B(self):
        dB_dt = np.diff(self.field_E, axis=-1) / self.dt
        self.field_B[..., 0] = -dB_dt[..., 0]
        self.field_B[..., 1:] = -integrate.cumtrapz(
            dB_dt, dx=self.dt, initial=0, axis=-1
        )
        self.field_B = np.where(np.isnan(self.field_B), np.nanmin(self.field_B), self.field_B)
    

    def calc_next(self):
       
        for part in self.particles:
            # Vérifier si les coordonnées de la particule restent dans les limites du monde simulé
            if 0 <= part.x < self.size and 0 <= part.y < self.size and 0 <= part.z < self.size:
                # Si les coordonnées sont valides, calculer la prochaine position de la particule
                part.calc_next(self.field_E, self.field_B, self.size, self.dt)
            else:
                # Si les coordonnées sont invalides, ignorer la mise à jour de la particule
                print(f"Attention : Les coordonnées de la particule sont hors des limites du monde simulé,  {part.x}{part.y}{part.z}.")
        self.temps += self.dt
        self.calc_E()
        self.calc_B()
    def create_animation(
        self, fps, total_simulation_time, total_animation_time, animation_type, output_folder_name
    ):
        current_time = datetime.datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Simulation_{self.size}_{self.cell_size}_{self.dt}_{fps}fps_{current_time_str}.mp4"

        output_folder = os.path.join(
            output_folder_name,
            f"Simulation_{self.size}_{self.cell_size}_{self.dt}_{fps}fps_{current_time_str}",
        )

        os.makedirs(output_folder, exist_ok=True)
        animation_output_path = os.path.join(output_folder, filename)

        console_output_path = os.path.join(
            output_folder,
            f"Simulation_{self.size}_{self.cell_size}_{self.dt}_{fps}fps_{current_time_str}_console.txt",
        )   

        with open(console_output_path, "w") as console_file:
            sys.stdout = console_file
            print(f"Paramètres de simulation : size={self.size}, cell_size={self.cell_size}, dt={self.dt}")
            print(f"Nombre de particules : {len(self.particles)}")
            print("Paramètres initiaux des particules :")
            for i, part in enumerate(self.particles):
                print(f"Particule {i+1}: x={part.x}, y={part.y}, z={part.z}, charge={part.charge}, masse={part.mass}")
            print(f"Type d'animation : {animation_type}")
            print(f"FPS de l'animation : {fps}")
            print(f"Durée totale de la simulation : {total_simulation_time} s")
            print(f"Durée totale de l'animation : {total_animation_time} s")
            print()
            sys.stdout = sys.__stdout__
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        simulation_time = 0  # Initialisation du temps de simulation

        def update(frame):
            nonlocal simulation_time  # Utilisation de la variable simulation_time déclarée en dehors de la fonction

            ax.clear()
            ax.set_xlim(0, self.size)
            ax.set_ylim(0, self.size)
            ax.set_zlim(0, self.size)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"Simulation - Temps écoulé (simulation): {simulation_time} s")  # Utilisation de simulation_time

            if animation_type == "P":
                self.plot_particle_positions(ax)
            elif animation_type in ["E", "B", "TOTAL"]:
                self.plot_fields(ax, field_type=animation_type)

            self.calc_next()
            simulation_time += self.dt  # Incrémentation du temps de simulation à chaque trame

        total_simulation_frames = int(total_simulation_time / self.dt)  # Utilisation de self.dt pour le temps de simulation
        total_animation_frames = int(total_animation_time / self.dt)  # Utilisation de self.dt pour le temps d'animation
        fps_interval = 1000 / fps

        with tqdm.tqdm(
            total=total_animation_frames, desc="Calcul de l'animation", unit="frames"
        ) as progress_bar:

            def update_with_progress(frame):
                update(frame)
                progress_bar.update(1)

            ani = animation.FuncAnimation(
                fig,
                update_with_progress,
                frames=total_animation_frames,
                interval=fps_interval,
            )

            ani.save(animation_output_path, writer="ffmpeg", fps=fps)

            print(f"L'animation a été enregistrée sous {animation_output_path}")
            print(f"Les informations sont également enregistrées dans {console_output_path}")

    def plot_particle_positions(self, ax):
        for part in self.particles:
            ax.scatter(part.x, part.y, part.z, c="r", marker="o")

    def plot_fields(self, ax, field_type, window_size=1):
        if field_type == "E":
            field = self.field_E
            field_label = "Champ électrique"
        elif field_type == "B":
            field = self.field_B
            field_label = "Champ magnétique"
        elif field_type == "TOTAL":
            field = self.field_E + self.field_B
            field_label = "Champ total (E + B)"
        else:
            raise ValueError("Type de champ non valide. Utilisez 'E', 'B' ou 'TOTAL'.")

        shape = field.shape[:-1]
        grid_size = [np.arange(0, s, self.cell_size) for s in shape]

        x_coords, y_coords, z_coords = np.meshgrid(*grid_size, indexing="ij")

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    i_min = max(0, i - window_size)
                    i_max = min(shape[0], i + window_size + 1)
                    j_min = max(0, j - window_size)
                    j_max = min(shape[1], j + window_size + 1)
                    k_min = max(0, k - window_size)
                    k_max = min(shape[2], k + window_size + 1)

                    window_field = field[i_min:i_max, j_min:j_max, k_min:k_max]
                    avg_field = np.mean(window_field, axis=(0, 1, 2))

                    ax.quiver(
                        x_coords[i, j, k],
                        y_coords[i, j, k],
                        z_coords[i, j, k],
                        avg_field[0],
                        avg_field[1],
                        avg_field[2],
                        length=0.5,
                        normalize=True,
                        color='b',
                    )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.zaxis.label.set_color("white")

        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.tick_params(axis="z", colors="white")


# Créer une instance de la classe World
w = World(70, 1, 0.01)  # Taille du monde, taille des cells, dt -(delta t)
w.add_part(Particle(4,4,4, 1*const.charge_electron , 1*const.masse_electron,1))
'''w.add_part(Particle(6,4,4, 100000000*const.charge_electron , 100000000*const.masse_electron))'''
fps = 10
duree_simulation = 0.01
duree_animation = 1
clear = False




def crf():
    folder_path = "Résultats"
    if not os.path.isdir(folder_path):
        print(f"Le chemin spécifié '{folder_path}' n'est pas un dossier existant.")
        return

    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        print(f"Le dossier '{folder_path}' a été vidé avec succès.")
    except Exception as e:
        print(f"Erreur lors de la suppression du contenu du dossier '{folder_path}': {e}")


if clear == True:
    crf()
else : 
    # Appeler la méthode create_animation avec les arguments spécifiés
    w.create_animation(
        fps, duree_simulation, duree_animation, output_folder_name="Résultats", animation_type="P" 
    )