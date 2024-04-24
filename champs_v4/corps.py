import constants as const
import numpy as np

from math import sqrt

class Particle:
    def __init__(self, x: float,  y: float, z: float, charge: float, mass: float, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,dim: int = 0) -> None:
        
        self.x: float = np.float64(x) * (10**dim)
        self.y: float = np.float64(y) * (10**dim)
        self.z: float = np.float64(z) * (10**dim)
        
        self.vx: float = np.float64(vx) 
        self.vy: float = np.float64(vy)
        self.vz: float = np.float64(vz)
        
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
        print( f"fX = {Fx} ex = {ex}, bx {bx}, bx * self.vx {bx * self.vx}, self.vx = {self.vx} ax = {ax}")
        # Vérifier si les valeurs sont NaN
        if np.isnan(ax) or np.isnan(ay) or np.isnan(az):
            print(F"Une des accélérations calculées est NaN. fX = {Fx}")
        
        self.x = ax * dt
        self.y = ay * dt  
        self.z = az * dt

        
        """ # Étape 1: Calcul des coefficients RK4
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
        """

        """
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
