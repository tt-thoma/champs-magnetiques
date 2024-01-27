import constants as const
import numpy as np

from math import sqrt

class Particle:
    def __init__(
        self, x: int, y: int, charge: float, vx: float = 0.0, vy: float = 0.0
    ) -> None:
        self.x: float = x
        self.y: float = y
        self.vx: float = vx
        self.vy: float = vy

        self.charge: float = charge

        if charge < 0:
            self.mass: float = 9.11 * (10**-27)
        elif charge > 0:
            self.mass: float = 1.673 * (10**-27)
        else:
            self.mass: float = 1.675 * (10**-27)

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
            
    def _calc(self, world, size: int) -> None:
        points = np.arange(size**2)
        x = points % size
        y = points // size

        mask = (x != self.x) | (y != self.y)
        vec_x = np.where(mask, x - self.x, 0).astype(float)
        vec_y = np.where(mask, y - self.y, 0).astype(float)

        vec_norm = np.sqrt(vec_x**2 + vec_y**2) + 1e-9

        force = abs(np.where(mask, abs((const.k * self.charge) / vec_norm**2), 0))

        uni_mult = force / vec_norm
        vec_x *= uni_mult * np.sign(self.charge)
        vec_y *= uni_mult * np.sign(self.charge)

        world[:, 0] += force
        world[:, 1] += vec_x
        world[:, 2] += vec_y

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

