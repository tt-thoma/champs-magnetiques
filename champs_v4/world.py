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
from utils import print_debug

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
            px, py, pz = int(part.x/self.cell_size), int(part.y/self.cell_size), int(part.z/self.cell_size)
            
            vec_pos = np.stack(
                (x_coords - part.x, y_coords - part.y, z_coords - part.z), axis=-1
            )
            distance = np.linalg.norm(vec_pos, axis=-1)
                
            # Exclure la contribution si la distance est nulle (particule à la même position)
            # valid_distance = np.zeros_like(self.field_E[:2])
            # valid_distance[px, py, pz] = 1
            # valid_distance = valid_distance == 0

            # Utiliser np.newaxis pour étendre la dimension sur laquelle nous appliquons le masque booléen
            E_norm = const.k * part.charge / ((distance**2)+1e-20).astype(np.float64)
            E_vec = E_norm[..., np.newaxis] * vec_pos / distance[..., np.newaxis]
            E_vec[px, py, pz] = [0,0,0]
            
            # Appliquer le masque uniquement sur les dimensions appropriées
            # total_E += np.where(distance != 0, E_vec, 0)
            total_E += E_vec
            
        self.field_E = total_E
        if np.isnan(self.field_E).any():
            print_debug(f"Le champ électrique contient des valeurs NaN après le calcul.")

    def calc_B(self):
        dB_dt = np.diff(self.field_E, axis=-1) / self.dt
        self.field_B[..., 0] = -dB_dt[..., 0]
        self.field_B[..., 1:] = -integrate.cumtrapz(
            dB_dt, dx=self.dt, initial=0, axis=-1
        )
        self.field_B = np.where(
            np.isnan(self.field_B), np.nanmin(self.field_B), self.field_B
        )

    def calc_next(self):
        for part in self.particles:
            # Vérifier si les coordonnées de la particule restent dans les limites du monde simulé
            if (
                0 <= part.x < self.size
                and 0 <= part.y < self.size
                and 0 <= part.z < self.size
            ):
                # Si les coordonnées sont valides, calculer la prochaine position de la particule
                part.calc_next(self.field_E, self.field_B, self.size, self.dt, self.cell_size)
            else:
                # Si les coordonnées sont invalides, ignorer la mise à jour de la particule
                print_debug(
                    f"Attention : Les coordonnées de la particule sont hors des limites du monde simulé,  x = {part.x} / y = {part.y} / z = {part.z}."
                )
        self.temps += self.dt
        # self.temps = round(self.temps, int(1/self.dt))
        
        self.calc_E()
        self.calc_B()
        
       

    def create_animation(
        self,
        total_simulation_time,
        total_animation_time,
        animation_type,
        output_folder_name,
        cell_size_reduction,
        v,
        r,
        particule_visualisation,
    ):
        current_time = datetime.datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Simulation_{self.size}_{self.cell_size}_{self.dt}_fps_{current_time_str}.mp4"

        output_folder = os.path.join(
            output_folder_name,
            f"Simulation_{self.size}_{self.cell_size}_{self.dt}_fps_{current_time_str}",
        )

        os.makedirs(output_folder, exist_ok=True)
        animation_output_path = os.path.join(output_folder, filename)

        console_output_path = os.path.join(
            output_folder,
            f"Simulation_{self.size}_{self.cell_size}_{self.dt}_fps_{current_time_str}_console.txt",
        )

        with open(console_output_path, "w") as console_file:
            sys.stdout = console_file
            print_debug(
                f"Paramètres de simulation : size={self.size}, cell_size={self.cell_size}, dt={self.dt}"
            )
            print_debug(f"Nombre de particules : {len(self.particles)}")
            print_debug("Paramètres initiaux des particules :")
            for i, part in enumerate(self.particles):
                print_debug(
                    f"Particule {i+1}: x={part.x}, y={part.y}, z={part.z}, charge={part.charge}, masse={part.mass}"
                )
            print_debug(f"Type d'animation : {animation_type}")
            
            print_debug(f"Durée totale de la simulation : {total_simulation_time} s")
            print_debug(f"Durée totale de l'animation : {total_animation_time} s")
            print_debug()
            sys.stdout = sys.__stdout__
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        #ax.view_init(azim=r, elev=v)
        simulation_time = 0  # Initialisation du temps de simulation

        def update():
            nonlocal simulation_time  # Utilisation de la variable simulation_time déclarée en dehors de la fonction

            ax.clear()
            ax.set_xlim(0, self.size-1/self.cell_size)
            ax.set_ylim(0, self.size-1/self.cell_size)
            ax.set_zlim(0, self.size-1/self.cell_size)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            
            ax.set_title(
                f"Simulation - Temps écoulé (simulation): {w.temps} s"
            )

            if animation_type == "P":
                self.plot_particle_positions(ax)
            elif animation_type in ["E", "B", "T"]:
                self.plot_fields(ax, field_type=animation_type, cell_size_reduction = cell_size_reduction)
            
            if particule_visualisation == True and animation_type in ["E", "B", "T"]:
                self.plot_fields(ax, field_type=animation_type, cell_size_reduction = cell_size_reduction)
                self.plot_particle_positions(ax)
            self.calc_next()
            simulation_time += self.dt # Incrémentation du temps de simulation à chaque trame

        # fps = nombre de frame / temps 
        total_animation_frames = int(
            total_simulation_time / self.dt
        )  # Utilisation de self.dt pour le temps d'animation
        
        animation_simulation_interval = (total_simulation_time / total_animation_frames) * total_animation_time*10**(3-np.log10(duree_simulation))
        
        with tqdm.tqdm(
            total=total_animation_frames, desc="Calcul de l'animation", unit="frames"
        ) as progress_bar:

            def update_with_progress(frame):
                update()
                progress_bar.update(1)

            ani = animation.FuncAnimation(
                fig,
                update_with_progress,
                frames=total_animation_frames,
                interval=animation_simulation_interval,
            )

            ani.save(animation_output_path, writer="ffmpeg") #, fps=fps)

            print_debug(f"L'animation a été enregistrée sous {animation_output_path}")
            print_debug(
                f"Les informations sont également enregistrées dans {console_output_path}"
            )
            print_debug(f"{total_animation_time=}")
            print_debug("#######", animation_simulation_interval)

    def plot_particle_positions(self, ax):
        for part in self.particles:
            ax.scatter(part.x, part.y, part.z, c="r", marker="o")
    def plot_fields(self, ax, field_type, cell_size_reduction):
        if field_type == "E":
            field = self.field_E
            field_label = "Champ électrique"
        elif field_type == "B":
            field = self.field_B
            field_label = "Champ magnétique"
        elif field_type == "T":
            field = self.field_E + self.field_B
            field_label = "Champ total (E + B)"
        else:
            raise ValueError("Type de champ non valide. Utilisez 'E', 'B' ou 'TOTAL'.")

        shape = field.shape[:-1]

        grid_size = np.arange(0, shape[0] * self.cell_size, self.cell_size)
        x_coords, y_coords, z_coords = np.meshgrid(grid_size, grid_size, grid_size, indexing="ij")

        # Réduire la taille de la grille
        reduced_shape = (shape[0] // cell_size_reduction, shape[1] // cell_size_reduction, shape[2] // cell_size_reduction)
        x_coords_reduced = x_coords[::cell_size_reduction, ::cell_size_reduction, ::cell_size_reduction]
        y_coords_reduced = y_coords[::cell_size_reduction, ::cell_size_reduction, ::cell_size_reduction]
        z_coords_reduced = z_coords[::cell_size_reduction, ::cell_size_reduction, ::cell_size_reduction]

        # Moyenne des vecteurs de champ dans les cellules
        averaged_field = np.mean(field.reshape((reduced_shape[0], cell_size_reduction, reduced_shape[1], cell_size_reduction, reduced_shape[2], cell_size_reduction, 3)), axis=(1, 3, 5))

        # Calcul de la norme du champ moyenné
        norm_values = np.linalg.norm(averaged_field, axis=3)
        norm = plt.Normalize(vmin=norm_values.min(), vmax=norm_values.max())
        norm_values_normalized = norm(norm_values)
        colors = plt.cm.RdBu(1 - norm_values_normalized.ravel())
        
        alphas = 0.5 + 0.5 * norm_values_normalized.ravel()

        ax.quiver(
            x_coords_reduced,
            y_coords_reduced,
            z_coords_reduced,
            averaged_field[..., 0],
            averaged_field[..., 1],
            averaged_field[..., 2],
            length=(1/self.cell_size)*cell_size_reduction,
            normalize=True,
            colors=colors,
            alpha=alphas
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

        ax.set_facecolor('black')

        ax.grid(color='white')
"""
        if not hasattr(self, 'colorbar_created'):
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu, norm=plt.Normalize(vmin=norm_values.min(), vmax=norm_values.max()))
            sm.set_array([])
            self.colorbar = plt.colorbar(sm, ax=ax, label='Norme du champ', pad=0.05)  # Ajustez le pad selon vos besoins
            self.colorbar_created = True"""
        
#----Temps-----
dt = 0.01 #s    
duree_simulation = 1
duree_animation = 10
#---boul------
clear = False
simulation = True
#----taille----
taille_du_monde = 10 # m
taille_des_cellules = 1 # m
cell_size_reduction = 1 #cell
#pdv axe degres
r = 45
v = 45

type_aniamtion = "E"
particule_visualisation = True
# fps = 30




# Créer une instance de la classe World
w = World(taille_du_monde, taille_des_cellules, dt)  # Taille du monde, taille des cells, dt -(delta t)
w.add_part(Particle(4, 5 ,5,  1* const.charge_electron, 1 * const.masse_electron,dim =0 ))
w.add_part(Particle(4 ,3,4, -1* const.charge_electron, 1 * const.masse_electron,dim = 0))



def crf():
    folder_path = "Résultats"
    if not os.path.isdir(folder_path):
        print_debug(f"Le chemin spécifié '{folder_path}' n'est pas un dossier existant.")
        return

    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        print_debug(f"Le dossier '{folder_path}' a été vidé avec succès.")
    except Exception as e:
        print_debug(
            f"Erreur lors de la suppression du contenu du dossier '{folder_path}': {e}"
        )


if clear == True:
    crf()
if simulation:
    w.create_animation(
        
        duree_simulation,
        duree_animation,
        output_folder_name="Résultats",
        animation_type= type_aniamtion,
        cell_size_reduction= cell_size_reduction,
        r = r,
        v = v,
        particule_visualisation = particule_visualisation,
        
    )
