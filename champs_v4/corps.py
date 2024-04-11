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
        

    def calc(self, world, size: int) -> None:
       pass

    def calc_next(self, world, size, dt):
        id = int(self.y) * size + int(self.x)
        Fx = world[id, 1] * self.charge
        Fy = world[id, 2] * self.charge
        accel_x = Fx / self.mass
        accel_y = Fy / self.mass
        k1_vx = dt * accel_x
        k1_vy = dt * accel_y
        k1_x = dt * self.vx
        k1_y = dt * self.vy

        k2_vx = dt * (accel_x + 0.5 * k1_vx)
        k2_vy = dt * (accel_y + 0.5 * k1_vy)
        k2_x = dt * (self.vx + 0.5 * k1_vx)
        k2_y = dt * (self.vy + 0.5 * k1_vy)

        k3_vx = dt * (accel_x + 0.5 * k2_vx)
        k3_vy = dt * (accel_y + 0.5 * k2_vy)
        k3_x = dt * (self.vx + 0.5 * k2_vx)
        k3_y = dt * (self.vy + 0.5 * k2_vy)

        k4_vx = dt * (accel_x + k3_vx)
        k4_vy = dt * (accel_y + k3_vy)
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

