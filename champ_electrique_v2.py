# -*- encoding: utf-8 -*-
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt

c_coulomb = 8.854 * (10**-12)
charge_unit = 10 * 1.602 * 10**-19
unit_time = 10**-3
m2mm = 10**-3
mm2m = 10**3


class Particle:
    def __init__(
        self, x: int, y: int, charge: float, vx: float = 0.0, vy: float = 0.0
    ) -> None:
        self.x: float = x
        self.y: float = y
        self.vx: float = vx / unit_time * m2mm
        self.vy: float = vy / unit_time * m2mm

        self.charge: float = charge

        if charge < 0:
            self.mass: float = 9.11 * (10**-31)
        elif charge > 0:
            self.mass: float = 1.673 * (10**-27)
        else:
            self.mass: float = 1.675 * (10**-27)
        self.mass *= charge

    def calc(self, world, size):
        for point in enumerate(world):
            x = point[0] % size
            y = int(point[0] / size)

            if x == self.x and y == self.y:
                world[point[0], :] = 0
                continue

            vec_x = x - self.x
            vec_y = y - self.y

            vec_norm = sqrt(vec_x**2 + vec_y**2)

            force = (c_coulomb * self.charge) / vec_norm
            uni_mult = force**2 / vec_norm
            vec_x *= uni_mult
            vec_y *= uni_mult

            og_force = world[point[0]]
            world[point[0]] = force + og_force, vec_x, vec_y

    def calc_next(self, world, size):
        id = round(self.x) + size * round(self.y)

        # Accélération
        accel_x = world[id, 1] / self.mass
        accel_y = world[id, 2] / self.mass

        # Nouveau vecteur vitesse
        self.vx = self.vx + accel_x / unit_time
        self.vy = self.vy + accel_y / unit_time

        # Nouvelle position
        self.x = self.x + self.vx * unit_time
        self.y = self.y + self.vy * unit_time


class World:
    def __init__(self, size: int) -> None:
        self.size: int = size
        # Force, vecX, vecY
        self.world = np.zeros((size, 3))
        self.particles: list[Particle] = []

    def add_part(self, part: Particle) -> None:
        self.particles.append(part)

    def calc(self) -> None:
        for part in self.particles:
            part.calc(self.world, self.size)
            part.calc_next(self.world, self.size)

    def get_pos(self):
        img = np.zeros((self.size, self.size))
        for part in self.particles:
            x, y = round(part.x), round(part.y)
            if x > self.size or y > self.size:
                continue
            if x < 0 or y < 0:
                continue

            print(f"x: {x}, y: {y}")
            img[x, y] = 1

        return img

    def get_np(self):
        pass


world = World(100)
world.add_part(Particle(40, 40, 1 * (10**25) * charge_unit))
world.add_part(Particle(60, 60, -1 * (10**25) * charge_unit))

for _ in range(10):
    image = world.get_pos()
    plt.imshow(image)
    plt.show()
    world.calc()
