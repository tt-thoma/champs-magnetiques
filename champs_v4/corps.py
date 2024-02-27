import constants as const
import numpy as np

from math import sqrt

class Particle:
    def __init__(
        self, x: int, y: int, charge: float, vx: float = 0.0, vy: float = 0.0 , ax: float = 0.0 , ay: float = 0.0
    ) -> None:
        self.x: float = x
        self.y: float = y
        
        self.vx: float = vx
        self.vy: float = vy
        self.ax: float = ax
        self.ay: float = ay
        self.charge: float = charge
        
        if charge < 0:
            self.mass: float = const.charge_electron
        elif charge > 0:
            self.mass: float = const.charge_proton
        

    
    def calc_E(self, world_E: np.ndarray):
        shape = world_E.shape
        for y in range(shape[1]):
            for x in range(shape[0]):
                vec_x = x - self.x
                vec_y = y - self.y
                vec_norm2 = vec_x**2 + vec_y**2 + 1e-9
                vec_norm = sqrt(vec_norm2)
                
                force_norm = abs(const.k*self.charge)/vec_norm2
                uni_multi = force_norm / vec_norm
                vec_x *= uni_multi * np.sign(self.charge)
                vec_y *= uni_multi * np.sign(self.charge)
                
                world_E[x, y] += vec_x, vec_y
     

    def calc_next(self, world_E, world_B, size, dt):
        ex, ey = world_E[self.x, self.y]
        bx, by = world_B[self.x, self.y]
        
        Fx = self.charge * (ex + bx * self.ax)
        Fy = self.charge * (ey + by * self.ay)
        
        self.ax = Fx / self.mass
        self.ay = Fy / self.mass
        k1_xx²
        k1_vx = dt * self.ax
        k1_vy = dt * self.ay
        k1_x = dt * self.vx
        k1_y = dt * self.vy

        k2_vx = dt * (self.ax + 0.5 * k1_vx)
        k2_vy = dt * (self.ay + 0.5 * k1_vy)
        k2_x = dt * (self.vx + 0.5 * k1_vx)
        k2_y = dt * (self.vy + 0.5 * k1_vy)

        k3_vx = dt * (self.ax + 0.5 * k2_vx)
        k3_vy = dt * (self.ay + 0.5 * k2_vy)
        k3_x = dt * (self.vx + 0.5 * k2_vx)
        k3_y = dt * (self.vy + 0.5 * k2_vy)

        k4_vx = dt * (self.ax + k3_vx)
        k4_vy = dt * (self.ay + k3_vy)
        k4_x = dt * (self.vx + k3_vx)
        k4_y = dt * (self.vy + k3_vy)

        new_vx = self.vx + (k1_vx + 2 * k2_vx + 2 * k3_vx + k4_vx) / 6
        new_vy = self.vy + (k1_vy + 2 * k2_vy + 2 * k3_vy + k4_vy) / 6
        new_x = self.x + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        new_y = self.y + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6

        if new_x <= 0 or new_x >= size:
            new_vx = -new_vx * 0.5
        else:
            self.x = new_x

        if new_y <= 0 or new_y >= size:
            new_vy = -new_vy * 0.5
        else:
            self.y = new_y

        self.vx = new_vx
        self.vy = new_vy

