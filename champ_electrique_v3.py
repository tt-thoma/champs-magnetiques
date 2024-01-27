# -*- encoding: utf-8 -*-
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import random


# c_coulomb = 8.854 * (10**-12)
c_coulomb = 8.854 * (10**-12)
charge_unit = 10 * 1.602 * 10**-19
unit_time = 0.001


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

    def calc(self, world, size: int) -> None:
        points = np.arange(size**2)
        x = points % size
        y = points // size

        mask = (x != self.x) | (y != self.y)
        vec_x = np.where(mask, x - self.x, 0).astype(float)
        vec_y = np.where(mask, y - self.y, 0).astype(float)

        vec_norm = np.sqrt(vec_x**2 + vec_y**2) + 1e-9

        force = abs(np.where(mask, abs((c_coulomb * self.charge) / vec_norm**2), 0))

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

    def check_collision(self, other):
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2) <= sqrt(2)


class World:
    def __init__(self, size: int, dt: float) -> None:
        self.size: int = size
        self.dt: float = dt
        # Force, vecX, vecY
        self.world = np.zeros((size**2, 3))
        self.particles: list[Particle] = []

    def add_part(self, part: Particle) -> None:
        self.particles.append(part)

    def calc(self):
        for part in self.particles:
            part.calc(self.world, self.size)
        for i in range(len(self.particles)):
            for j in range(i + 1, len(self.particles)):
                if self.particles[i].check_collision(self.particles[j]):
                    self.particles[i].vx = 0
                    self.particles[i].vy = 0
                    self.particles[j].vx = 0
                    self.particles[j].vy = 0
            self.particles[i].calc_next(self.world, self.size, self.dt)

    def get_pos(self):
        img = np.zeros((self.size, self.size))
        for part in self.particles:
            x, y = round(part.x), round(part.y)
            if x >= self.size or y >= self.size:
                continue
            if x < 0 or y < 0:
                continue

            img[-y, x] = 1
        return img

    def animate(self, ax):
        self.calc()
        self.img.set_array(self.get_pos())

        _x, _y = np.meshgrid(np.arange(self.size), np.arange(self.size))
        # Calc unit vectors
        unit = np.sqrt(self.world[:, 1] ** 2 + self.world[:, 2] ** 2)
        unit_x = self.world[:, 1] / unit
        unit_y = self.world[:, 1] / unit

        self.q = ax.quiver(_x, _y, unit_x, unit_y)

        return [self.img, self.q]

    def show_field(self):
        X, Y = np.meshgrid(np.arange(self.size), np.arange(self.size))
        fig, ax = plt.subplots()
    
        # Calculer la norme des vecteurs
        magnitude = np.sqrt(self.world[:, 1]**2 + self.world[:, 2]**2)
    
        # Normaliser la norme des vecteurs
        normalized_magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
    
        # Créer des vecteurs unitaires
        ux = self.world[:, 1] / magnitude
        uy = self.world[:, 2] / magnitude
    
        # Créer une échelle de couleurs basée sur la norme normalisée
        colors = normalized_magnitude
    
        # Créer le graphique
        Q = ax.quiver(X, Y, ux, uy, colors, angles='xy', scale_units='xy', scale=1)
    
        # Ajuster l'échelle des vecteurs pour les rendre plus visibles
        Q.set_UVC(ux/magnitude, uy/magnitude)
    
        # Ajuster l'échelle de couleurs pour qu'elle corresponde à la plage de valeurs de la norme normalisée
        plt.colorbar(Q, ax=ax, label='Norme du vecteur', extend='both')
    
        plt.show()




    def create_animation(self):
        fig, ax = plt.subplots()
        ax.axis("on")
        self.img = ax.imshow(self.get_pos(), animated=True)

        ani = FuncAnimation(fig, self.animate, frames=1, interval=1, blit=True)
        plt.show()


dt = 0.01
world = World(10, dt)
"""
for i in range(2):
    p = Particle(
        random.randint(1, 98),
        random.randint(1, 98),
        random.randint(-10, 10) * 50000 * charge_unit,
    )
    world.add_part((p))
"""
world.add_part(Particle(2, 2, 500000 * charge_unit))
world.add_part(Particle(3, 8, 500000 * charge_unit))
world.particles[0].calc(world.world, 10)
world.particles[1].calc(world.world, 10)


world.show_field()
# world.create_animation()
