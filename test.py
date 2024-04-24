import numpy as np

from scipy.constants import epsilon_0, e, m_e, m_p
from math import pi
# Constante de Coulomb
k = 1 / (4 * epsilon_0 * pi)
# Charge d'un électron
charge_electron = -e
# Charge d'un proton
charge_proton = e
# Masse d'un électron
masse_electron = m_e
# Masse d'un proton
masse_proton = m_p


import numpy as np

from math import sqrt

class Particle:
    def __init__(self, x: float,  y: float, z: float, charge: float, mass: float, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,dim: int = 0) -> None:
        
        self.x: float = np.float64(x) * (10**dim)
        self.y: float = np.float64(y) * (10**dim)
        self.z: float = np.float64(z) * (10**dim)
        
        self.vx: float = np.float64(vx) * (10**dim)
        self.vy: float = np.float64(vy) * (10**dim)
        self.vz: float = np.float64(vz) * (10**dim)
        
        self.charge: float = np.float64(charge)
        self.mass = np.float64(mass)
        

    def calc_next(self, world_E, world_B, size, dt):
        
        ex, ey, ez = world_E[int(self.x), int(self.y), int(self.z), :]
        bx, by, bz = world_B[int(self.x), int(self.y), int(self.z), :]
        
        Fx = self.charge * (ex + bx * self.vx)
        Fy = self.charge * (ey + by  * self.vy)
        Fz = self.charge * (ez + bz * self.vz)
        
        ax = Fx / self.mass
        ay = Fy / self.mass 
        az = Fz / self.mass
        
        # Vérifier si les valeurs sont NaN
        if np.isnan(ax) or np.isnan(ay) or np.isnan(az):
            print("Une des accélérations calculées est NaN.")
    
        
        k1_vx = dt * ax
        k1_vy = dt * ay
        k1_vz = dt * az
        k1_x = dt * self.vx
        k1_y = dt * self.vy
        k1_z = dt * self.vz
            
        k2_vx = dt * (ax + 0.5 * k1_vx)
        k2_vy = dt * (ay + 0.5 * k1_vy)
        k2_vz = dt * (az + 0.5 * k1_vz)
        k2_x = dt * (self.vx + 0.5 * k1_vx)
        k2_y = dt * (self.vy + 0.5 * k1_vy)
        k2_z = dt * (self.vz + 0.5 * k1_vz)

        k3_vx = dt * (ax + 0.5 * k2_vx)
        k3_vy = dt * (ay + 0.5 * k2_vy)
        k3_vz = dt * (az + 0.5 * k2_vz)
        k3_x = dt * (self.vx + 0.5 * k2_vx)
        k3_y = dt * (self.vy + 0.5 * k2_vy)
        k3_z = dt * (self.vz + 0.5 * k2_vz)

        k4_vx = dt * (ax + k3_vx)
        k4_vy = dt * (ay + k3_vy)
        k4_vz = dt * (az + k3_vz)
        k4_x = dt * (self.vx + k3_vx)
        k4_y = dt * (self.vy + k3_vy)
        k4_z = dt * (self.vz + k3_vz)

        new_vx = self.vx + (k1_vx + 2 * k2_vx + 2 * k3_vx + k4_vx) / 6
        new_vy = self.vy + (k1_vy + 2 * k2_vy + 2 * k3_vy + k4_vy) / 6
        new_vz = self.vz + (k1_vz + 2 * k2_vz + 2 * k3_vz + k4_vz) / 6
        new_x = self.x + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        new_y = self.y + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
        new_z = self.z + (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6
        
        if np.isnan(new_vx) or np.isnan(new_vy) or np.isnan(new_vz):
            print("Une des nouvelles vitesses calculées est NaN.")
        
        
        if new_x <= 0 or new_x >= size:
            new_vx = -new_vx * 0.5
        else:
            self.x = new_x

        if new_y <= 0 or new_y >= size:
            new_vy = -new_vy * 0.5
        else:
            self.y = new_y

        if new_z <= 0 or new_z >= size:
            new_vz = -new_vz * 0.5
        else:
            self.z = new_z

        self.vx = new_vx
        self.vy = new_vy
        self.vz = new_vz






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
            distance = np.where(distance == 0, 1e-9, distance)
            E_norm = k * part.charge / (distance**2).astype(np.float64)  # Calcul de la magnitude du champ électrique
            E_vec = E_norm[..., np.newaxis] * vec_pos / distance[..., np.newaxis]  # Calcul du vecteur champ électrique normalisé
            total_E += E_vec

        self.field_E = total_E
        if np.isnan(self.field_E).any():
            print(f"Le champ électrique contient des valeurs NaN après le calcul.")
            nan_indices = np.argwhere(np.isnan(self.field_E))
            print("Indices des valeurs NaN dans le champ électrique :", nan_indices)


    
    



