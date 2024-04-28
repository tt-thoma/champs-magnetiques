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
from utils import print_debug, debug
import random as rd
from corps import Particle
from concurrent.futures import ThreadPoolExecutor
import time
from scipy.fft import fftn, ifftn


class World:
    def __init__(self, size, cell_size, dt: float) -> None:
        self.size: int = size
        self.dt: float = dt
        self.cell_size = cell_size

        size_int = int(size / cell_size)
        self.field_E = np.zeros((size_int, size_int, size_int, 3), dtype=np.float64)
        self.field_E_fil = np.zeros((size_int, size_int, size_int, 3), dtype=np.float64)

        self.field_B = np.zeros((size_int, size_int, size_int, 3), dtype=np.float64)

        self.particles: list[Particle] = []

        self.temps = 0

    def add_part(self, part: Particle) -> None:
        self.particles.append(part)

    def add_fil(self, axe, position_x, position_y, norm_E, d: int):
        global I, duree_simulation
        n_p = d * int(self.size / self.cell_size)
        q = (I * duree_simulation) / n_p
        masse = ((q) / const.charge_electron) * const.masse_electron

        if axe == "x":
            self.field_E_fil[:, position_x, position_y, 0] = norm_E
            for i in range(n_p):
                x = (i * self.cell_size) / d

                y = position_x * self.cell_size
                z = position_y * self.cell_size
                self.add_part(Particle(x, y, z, const.charge_electron, masse, fil=True))

        elif axe == "y":
            self.field_E_fil[:, position_x, position_y, 0] = norm_E
            for i in range(n_p):
                x = position_x * self.cell_size
                y = (i * self.cell_size) / d
                z = position_y * self.cell_size
                self.add_part(Particle(x, y, z, const.charge_electron, masse, fil=True))
        elif axe == "z":
            self.field_E_fil[:, position_x, position_y, 0] = norm_E
            for i in range(n_p):
                x = position_x * self.cell_size
                y = position_y * self.cell_size
                z = i * self.cell_size
                self.add_part(Particle(x, y, z, const.charge_electron, masse, fil=True))

    def calc_E2(self) -> None:

        shape = self.field_E.shape
        x_coords, y_coords, z_coords = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
        )

        x_coords = x_coords * self.cell_size
        y_coords = y_coords * self.cell_size
        z_coords = z_coords * self.cell_size

        total_E = np.zeros_like(self.field_E)

        # Function to calculate electric field contribution from a particle
        def calculate_E(part):
            px, py, pz = (
                int(part.x / self.cell_size),
                int(part.y / self.cell_size),
                int(part.z / self.cell_size),
            )
            vec_pos = np.stack(
                (x_coords - part.x, y_coords - part.y, z_coords - part.z), axis=-1
            )
            distance = np.linalg.norm(vec_pos, axis=-1)

            # Avoid division by zero by adding a small epsilon value
            E_norm = const.k * part.charge / ((distance**2) + 1e-20).astype(np.float64)
            E_vec = E_norm[..., np.newaxis] * vec_pos / distance[..., np.newaxis]
            E_vec[px, py, pz] = [
                0,
                0,
                0,
            ]  # Zero out electric field at particle's position

            return E_vec

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            # Map particle calculation to executor and sum results
            results = executor.map(calculate_E, self.particles)
            total_E = np.sum(list(results), axis=0)

        self.field_E = total_E

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
            px, py, pz = (
                int(part.x / self.cell_size),
                int(part.y / self.cell_size),
                int(part.z / self.cell_size),
            )

            vec_pos = np.stack(
                (x_coords - part.x, y_coords - part.y, z_coords - part.z), axis=-1
            )
            distance = np.linalg.norm(vec_pos, axis=-1)

            # Utiliser np.newaxis pour étendre la dimension sur laquelle nous appliquons le masque booléen
            E_norm = const.k * part.charge / ((distance**2) + 1e-20).astype(np.float64)
            E_vec = E_norm[..., np.newaxis] * vec_pos / distance[..., np.newaxis]
            E_vec[px, py, pz] = [0, 0, 0]

            # Appliquer le masque uniquement sur les dimensions appropriées

            total_E += E_vec

        self.field_E = total_E

        if np.isnan(self.field_E).any():
            print_debug(
                f"Le champ électrique contient des valeurs NaN après le calcul."
            )

    def _calculate_properties_in_cell(self, particle: "Particle") -> tuple:
        """
        Calculate velocity vector and charge for a given particle.

        Args:
            particle (Particle): Particle for which to calculate properties.

        Returns:
            tuple: Tuple containing the velocity vector (vx, vy, vz) and charge of the particle.
        """
        velocity = (particle.vx, particle.vy, particle.vz)
        return velocity, particle.charge

    def calculate_properties_in_cells(self) -> tuple:
        """
        Calculate the average velocity vector and charge of particles in each cell.

        Returns:
            tuple: Tuple containing arrays of average velocity vector and charge in each cell.
        """
        # Initialize arrays to store the sum of velocity vector and charge for each cell
        cell_velocity_sum = np.zeros(
            (self.field_E.shape[0], self.field_E.shape[1], self.field_E.shape[2], 3),
            dtype=np.float64,
        )
        cell_charge_sum = np.zeros_like(self.field_E[..., 0], dtype=np.float64)
        cell_particle_count = np.zeros_like(self.field_E[..., 0], dtype=int)

        # Function to calculate velocity vector and charge for a particle and update cell sums
        def update_cell_properties(particle):
            cell_x = int(particle.x / self.cell_size)
            cell_y = int(particle.y / self.cell_size)
            cell_z = int(particle.z / self.cell_size)
            velocity, charge = self._calculate_properties_in_cell(particle)
            cell_velocity_sum[cell_x, cell_y, cell_z] += velocity
            cell_charge_sum[cell_x, cell_y, cell_z] += charge
            cell_particle_count[cell_x, cell_y, cell_z] += 1

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            # Submit tasks for each particle to update cell properties
            executor.map(update_cell_properties, self.particles)

        # Calculate average velocity vector and charge in each cell
        cell_average_velocity = cell_velocity_sum / cell_particle_count[..., np.newaxis]
        cell_average_charge = cell_charge_sum / cell_particle_count

        return cell_average_velocity, cell_average_charge

    def calculate_current_density(self):
        """
        Calculate the current density in each cell based on particle velocities and charges.

        Returns:
            numpy.ndarray: Array containing the current density in each cell.
        """
        # Get average velocity vector and charge in each cell
        V, Q = self.calculate_properties_in_cells()

        # Calculate current density
        J = (
            V * Q[..., np.newaxis]
        )  # Element-wise multiplication to get contributions from each particle
        J_total = np.sum(
            J, axis=(3)
        )  # Sum contributions from all particles in each cell

        return J_total

    def calculate_rotationnel_E_fft(self):
        shape = self.field_E.shape
        rotationnel_E = np.zeros_like(self.field_E)

        # Transformée de Fourier du champ électrique
        E_fft = fftn(self.field_E, axes=(0, 1, 2))

        # Définition du vecteur d'onde
        kx = np.fft.fftfreq(shape[0], d=self.cell_size)
        ky = np.fft.fftfreq(shape[1], d=self.cell_size)
        kz = np.fft.fftfreq(shape[2], d=self.cell_size)

        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")

        # Calcul du spectre du rotationnel en espace des vecteurs d'onde
        rotationnel_E_fft_x = 1j * (KY * E_fft[..., 2] - KZ * E_fft[..., 1])
        rotationnel_E_fft_y = 1j * (KZ * E_fft[..., 0] - KX * E_fft[..., 2])
        rotationnel_E_fft_z = 1j * (KX * E_fft[..., 1] - KY * E_fft[..., 0])

        # Concaténation pour obtenir le spectre du rotationnel
        rotationnel_E_fft = np.stack(
            (rotationnel_E_fft_x, rotationnel_E_fft_y, rotationnel_E_fft_z), axis=-1
        )

        # Transformée inverse de Fourier pour obtenir le rotationnel dans l'espace physique
        rotationnel_E = ifftn(rotationnel_E_fft, axes=(0, 1, 2)).real

        return rotationnel_E

    def calc_B(self):
        rotationnel_E = self.calculate_rotationnel_E_fft()
        # Méthode de Runge-Kutta d'ordre 4 (RK4) pour l'intégration temporelle
        k1 = -const.mu_0 * rotationnel_E
        k2 = -const.mu_0 * (rotationnel_E + 0.5 * self.dt * k1)
        k3 = -const.mu_0 * (rotationnel_E + 0.5 * self.dt * k2)
        k4 = -const.mu_0 * (rotationnel_E + self.dt * k3)

        self.field_B = self.field_B + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def calc_next(self):

        for part in self.particles:

            # Vérifier si les coordonnées de la particule restent dans les limites du monde simulé
            if (
                0 <= part.x < self.size
                and 0 <= part.y < self.size
                and 0 <= part.z < self.size
            ):
                # Si les coordonnées sont valides, calculer la prochaine position de la particule
                part.calc_next(
                    self.field_E,
                    self.field_B,
                    self.size,
                    self.dt,
                    self.cell_size,
                    fil=self.field_E_fil,
                )
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
        min_alpha,
        max_alpha,
    ):

        current_time = datetime.datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Simulation_{current_time_str}.mp4"

        output_folder = os.path.join(
            output_folder_name,
            f"Simulation_{current_time_str}",
        )

        os.makedirs(output_folder, exist_ok=True)
        animation_output_path = os.path.join(output_folder, filename)

        console_output_path = os.path.join(
            output_folder,
            f"Simulation_{current_time_str}_console.txt",
        )

        with open(console_output_path, "w") as console_file:
            sys.stdout = console_file
            print(
                f"Paramètres de simulation : size={self.size}, cell_size={self.cell_size}, dt={self.dt}"
            )
            print(f"Nombre de particules : {len(self.particles)}")
            print("Paramètres initiaux des particules :")
            for i, part in enumerate(self.particles):
                print(
                    f"Particule {i+1}: x={part.x}, y={part.y}, z={part.z}, charge={part.charge}, masse={part.mass}"
                )
            print(f"Type d'animation : {animation_type}")

            print(f"Durée totale de la simulation : {total_simulation_time} s")
            print(f"Durée totale de l'animation : {total_animation_time} s")

            sys.stdout = sys.__stdout__
        fig = plt.figure()
        fig = plt.figure(facecolor="black")
        ax = fig.add_subplot(111, projection="3d")

        ax.view_init(azim=r, elev=v)
        simulation_time = 0  # Initialisation du temps de simulation

        def update():
            nonlocal simulation_time  # Utilisation de la variable simulation_time déclarée en dehors de la fonction

            ax.clear()
            ax.set_xlim(0, self.size - (self.cell_size))
            ax.set_ylim(0, self.size - (self.cell_size))
            ax.set_zlim(0, self.size - (self.cell_size))
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_facecolor("black")

            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.zaxis.label.set_color("white")

            ax.tick_params(axis="x", colors="white")
            ax.tick_params(axis="y", colors="white")
            ax.tick_params(axis="z", colors="white")

            ax.grid(color="white")
            ax.set_title(f"Simulation - Temps écoulé (simulation): {w.temps} s")

            if animation_type == "P":
                self.plot_particle_positions(ax)
            elif animation_type in ["E", "B", "T"]:
                self.plot_fields(
                    ax,
                    field_type=animation_type,
                    cell_size_reduction=cell_size_reduction,
                    min_alpha=min_alpha,
                    max_alpha=max_alpha,
                )

            if particule_visualisation == True and animation_type in ["E", "B", "T"]:
                self.plot_fields(
                    ax,
                    field_type=animation_type,
                    cell_size_reduction=cell_size_reduction,
                    min_alpha=min_alpha,
                    max_alpha=max_alpha,
                )
                self.plot_particle_positions(ax)
            self.calc_next()
            ax.set_title(
                f"Simulation - Temps écoulé (simulation): {w.temps} s", color="white"
            )

        # fps = nombre de frame / temps
        total_animation_frames = int(
            total_simulation_time / self.dt
        )  # Utilisation de self.dt pour le temps d'animation

        animation_simulation_interval = (
            (total_simulation_time / total_animation_frames)
            * total_animation_time
            * 10 ** (3 - np.log10(duree_simulation))
        )

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

            ani.save(animation_output_path, writer="ffmpeg")  # , fps=fps)

            print_debug(f"L'animation a été enregistrée sous {animation_output_path}")
            print_debug(
                f"Les informations sont également enregistrées dans {console_output_path}"
            )
            print_debug(f"{total_animation_time=}")
            print_debug("#######", animation_simulation_interval)

    def plot_particle_positions(self, ax):
        for part in self.particles:
            if part.fil == True:
                color = "w"
            else:
                color = "r"

            ax.scatter(part.x, part.y, part.z, c=color, marker="o")

    def plot_fields(
        self,
        ax,
        field_type,
        cell_size_reduction,
        min_alpha,
        max_alpha,
    ):

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
        x_coords, y_coords, z_coords = np.meshgrid(
            grid_size, grid_size, grid_size, indexing="ij"
        )

        # Réduire la taille de la grille
        reduced_shape = (
            shape[0] // cell_size_reduction,
            shape[1] // cell_size_reduction,
            shape[2] // cell_size_reduction,
        )
        x_coords_reduced = x_coords[
            ::cell_size_reduction, ::cell_size_reduction, ::cell_size_reduction
        ]
        y_coords_reduced = y_coords[
            ::cell_size_reduction, ::cell_size_reduction, ::cell_size_reduction
        ]
        z_coords_reduced = z_coords[
            ::cell_size_reduction, ::cell_size_reduction, ::cell_size_reduction
        ]

        # Moyenne des vecteurs de champ dans les cellules
        averaged_field = np.mean(
            field.reshape(
                (
                    reduced_shape[0],
                    cell_size_reduction,
                    reduced_shape[1],
                    cell_size_reduction,
                    reduced_shape[2],
                    cell_size_reduction,
                    3,
                )
            ),
            axis=(1, 3, 5),
        )

        # Calcul de la norme du champ moyenné
        norm_values = np.linalg.norm(averaged_field, axis=3)
        norm = plt.Normalize(vmin=norm_values.min(), vmax=norm_values.max())
        norm_values_normalized = norm(norm_values)
        colors = plt.cm.RdBu(1 - norm_values_normalized.ravel())

        alphas = min_alpha + max_alpha * norm_values_normalized.ravel()

        ax.quiver(
            x_coords_reduced,
            y_coords_reduced,
            z_coords_reduced,
            averaged_field[..., 0],
            averaged_field[..., 1],
            averaged_field[..., 2],
            length=(1 / self.size * self.cell_size) * cell_size_reduction,
            normalize=True,
            colors=colors,
            alpha=alphas,
        )
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.zaxis.label.set_color("white")

        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.tick_params(axis="z", colors="white")


"""
        if not hasattr(self, 'colorbar_created'):
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu, norm=plt.Normalize(vmin=norm_values.min(), vmax=norm_values.max()))
            sm.set_array([])
            self.colorbar = plt.colorbar(sm, ax=ax, label='Norme du champ', pad=0.05)  # Ajustez le pad selon vos besoins
            self.colorbar_created = True"""

# ----Temps-----
dt = 1e-7  # s
duree_simulation = 1e-5  # s
duree_animation = 20  # s

# ---bool------
clear = False
simulation = True
debug = True
type_simulation = "fil"  # "fil" , "R" , ""

# ----taille----
taille_du_monde = 1  # m
taille_des_cellules = 0.1  # m
cell_size_reduction = 1  # cell
dimension = 0  # int

# random
nombres_de_particules = 2  # int

# fill
axe = "x"
position_x = 4 - 1  # cell
position_y = 4 - 1  # cell
I = 1  # A
U = 1  # V
densité = 2

# animation
type_aniamtion = "B"  # "P", "E" ,"B" ,"T"
particule_visualisation = True
min_alpha = 0.0005  # 0 - 1
max_alpha = 0.7  # 0 - 1
# pdv axe
r = 45  # 0 - 180 degrès
v = 45  # 0 - 180 degrès


# Créer une instance de la classe World
w = World(
    taille_du_monde, taille_des_cellules, dt
)  # Taille du monde, taille des cells, dt -(delta t)

# w.add_part(Particle(1.5 ,1,1, -1* const.charge_electron, 1 * const.masse_electron,dim = dimension))

# w.add_part(Particle(4,4,4,1,1))


def p_random(nombres_de_particules):
    for particule in range(nombres_de_particules):
        x = rd.random() * taille_du_monde * taille_des_cellules / 10
        y = rd.random() * taille_du_monde * taille_des_cellules / 10
        z = rd.random() * taille_du_monde * taille_des_cellules / 10
        q = rd.choice((-1, 1)) * const.charge_electron
        q = const.charge_electron
        mass = const.masse_electron

        w.add_part(Particle(x, y, z, q, mass))


def crf():
    folder_path = "Résultats"
    if not os.path.isdir(folder_path):
        print_debug(
            f"Le chemin spécifié '{folder_path}' n'est pas un dossier existant."
        )
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


if type_simulation == "R":
    print("R")
    p_random(nombres_de_particules)
elif type_simulation == "fil":
    E_norm = (I**2) / (U * const.epsilon_0)
    w.add_fil(axe, position_x, position_y, E_norm, densité)


if min_alpha + max_alpha > 1:
    raise ValueError("Alpha ,n'est pas encadré entre 0 et 1")
if clear == True:
    crf()
if simulation:
    w.create_animation(
        duree_simulation,
        duree_animation,
        output_folder_name="Résultats",
        animation_type=type_aniamtion,
        cell_size_reduction=cell_size_reduction,
        r=r,
        v=v,
        particule_visualisation=particule_visualisation,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
    )
