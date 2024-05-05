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
    def __init__(self, x: float,  y: float, z: float, charge: float, mass: float, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0, ax= 0.0 , ay=0.0 ,az=0.0 ,dim: int = 0) -> None:
        
        self.x: np.float64 = np.float64(x) * (10**dim)
        self.y: np.float64 = np.float64(y) * (10**dim)
        self.z: np.float64 = np.float64(z) * (10**dim)
        
        self.vx: np.float64 = np.float64(vx) 
        self.vy: np.float64 = np.float64(vy)
        self.vz: np.float64 = np.float64(vz)
        
        self.ax: np.float64 = np.float64(ax)
        self.ay: np.float64 = np.float64(ay)
        self.az: np.float64 = np.float64(az)
        
        self.charge: np.float64 = np.float64(charge)
        self.mass = np.float64(mass)
        self.U = 10
        self.I = 10
        
    def calc_next(self, world_E, world_B, size, dt):
        ex, ey, ez = world_E[int(self.x), int(self.y), int(self.z), :]
        
        print(f"Coordonnées: {self.x=};{self.y=};{self.z=}")
        print(f"Multiplicateur: {0}")
        
        print(f"Champ_e: {ex=};{ey=};{ez=}")
        bx, by, bz = world_B[int(self.x), int(self.y), int(self.z), :]
        print(f"Champ_b: {bx=};{by=};{bz=}")
        
        Fx = self.charge * (ex + bx * self.vx)
        Fy = self.charge * (ey + by  * self.vy)
        Fz = self.charge * (ez + bz * self.vz)
        
        self.ax += Fx / self.mass
        self.ay += Fy / self.mass 
        self.az += Fz / self.mass
        print( f"fX = {Fx} ex = {ex}, bx {bx}, bx * self.vx {bx * self.vx}, self.vx = {self.vx} ax = {self.ax}")
        # Vérifier si les valeurs sont NaN
        if np.isnan(self.ax) or np.isnan(self.ay) or np.isnan(self.az):
            print(F"Une des accélérations calculées est NaN. {Fx=}")
        
        self.vx += self.ax * dt
        self.vy += self.ay * dt
        self.vz += self.az * dt
        self.x += self.vx *dt
        self.y += self.vy *dt
        self.z += self.vz *dt

        
        """  # Étape 1: Calcul des coefficients RK4
        k1x = self.charge * (ex + bx * self.vx) / self.mass
        k1y = self.charge * (ey + by * self.vy) / self.mass
        k1z = self.charge * (ez + bz * self.vz) / self.mass
        
        k2x = self.charge * (ex + bx * (self.vx + 0.5 * dt * k1x)) / self.mass
        k2y = self.charge * (ey + by * (self.vy + 0.5 * dt * k1y)) / self.mass
        k2z = self.charge * (ez + bz * (self.vz + 0.5 * dt * k1z)) / self.mass
        
        k3x = self.charge * (ex + bx * (self.vx + 0.5 * dt * k2x)) / self.mass
        k3y = self.charge * (ey + by * (self.vy + 0.5 * dt * k2y)) / self.mass
        k3z = self.charge * (ez + bz * (self.vz + 0.5 * dt * k2z)) / self.mass
        
        k4x = self.charge * (ex + bx * (self.vx + dt * k3x)) / self.mass
        k4y = self.charge * (ey + by * (self.vy + dt * k3y)) / self.mass
        k4z = self.charge * (ez + bz * (self.vz + dt * k3z)) / self.mass
        
        # Étape 2: Calcul des valeurs intermédiaires
        vx_new = self.vx + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        vy_new = self.vy + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
        vz_new = self.vz + (dt / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
        
        # Étape 3: Calcul final
        self.x += dt * vx_new
        self.y += dt * vy_new
        self.z += dt * vz_new
        

        
        if np.isnan(vx_new) or np.isnan(vy_new) or np.isnan(vz_new):
         print(f"Une des nouvelles vitesses calculées est NaN. Fx = {Fx}")

        if self.x <= 0 or self.x >= size:
            vx_new = -vx_new * 0.5
        else:
            self.x += dt * vx_new

        if self.y <= 0 or self.y >= size:
            vy_new = -vy_new * 0.5
        else:
            self.y += dt * vy_new

        if self.z <= 0 or self.z >= size:
            vz_new = -vz_new * 0.5
        else:
            self.z += dt * vz_new

        self.vx = vx_new
        self.vy = vy_new
        self.vz = vz_new
          """    



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
            #print(f"vec pos = {vec_pos}")
            distance = np.linalg.norm(vec_pos, axis=-1)
            distance = np.where(distance == 0, 1e-9, distance)
            E_norm = (
                k * part.charge / (distance**2).astype(np.float64)
            )
            print(f"{E_norm=}")
              # Calcul de la magnitude du champ électrique
            E_vec = (
                E_norm[..., np.newaxis] * vec_pos / distance[..., np.newaxis]
            )  # Calcul du vecteur champ électrique normalisé
            print(f"{E_vec=}")
            total_E += E_vec

        self.field_E = total_E
        if np.isnan(self.field_E).any():
            print(f"Le champ électrique contient des valeurs NaN après le calcul.")
            nan_indices = np.argwhere(np.isnan(self.field_E))
            print("Indices des valeurs NaN dans le champ électrique :", nan_indices)
    def calc_next(self):
        self.calc_E()
        
        
        for part in self.particles:
            # Vérifier si les coordonnées de la particule restent dans les limites du monde simulé
            if (
                0 <= part.x < self.size
                and 0 <= part.y < self.size
                and 0 <= part.z < self.size
            ):
                # Si les coordonnées sont valides, calculer la prochaine position de la particule
                part.calc_next(self.field_E, self.field_B, self.size, self.dt)
            else:
                # Si les coordonnées sont invalides, ignorer la mise à jour de la particule
                print(
                    f"Attention : Les coordonnées de la particule sont hors des limites du monde simulé,  x = {part.x} / y = {part.y} / z = {part.z}."
                )
        self.temps += self.dt

    def solenoide(self,centre_x,centre_y,centre_z,longueur,axe,rayon,densité_de_spires,nombre_total):
        norm_E = (self.I**2)/(self.U*epsilon_0)
        nombre_de_spire = densité_de_spires*longueur
        for p in range(nombre_total):
            n = (np.pi*2) / nombre_de_spire 
            x = centre_x + rayon*np.cos(n*p)
            y = centre_y + rayon*np.sin(n*p)
            z = centre_z
            self.add_part(x,y,z,1,1)
           
        
    
w = World(10,1,1)


