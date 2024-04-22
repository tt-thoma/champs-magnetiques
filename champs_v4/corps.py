import constants as const
import numpy as np

from math import sqrt

class Particle:
    def __init__(self, x: float, y: float, z: float, charge: float, mass: float, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0) -> None:
        self.x: float = x
        self.y: float = y
        self.z: float = z
        
        self.vx: float = vx
        self.vy: float = vy
        self.vz: float = vz
        
        self.charge: float = charge
        self.mass = mass
        

    def calc_next(self, world_E, world_B, size, dt):
        print(self.mass)
        ex, ey, ez = world_E[int(self.x), int(self.y), int(self.z), :]
        bx, by, bz = world_B[int(self.x), int(self.y), int(self.z), :]
        
        Fx = self.charge * (ex + bx * self.vx)
        Fy = self.charge * (ey + by  * self.vy)
        Fz = self.charge * (ez + bz * self.vz)
        
        ax = Fx / self.mass
        ay = Fy / self.mass 
        az = Fz / self.mass
        
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
