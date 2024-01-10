# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:19:16 2024

@author: Tom, Thomas
"""

from math import sqrt
import numpy as np


# Constants
c_coulomb = 8.854 * (10**-12)
charge_unit = -10 * 1.602 * 10**19


class Particle:
    def __init__(self, x: int, y: int, charge: float):
        self.x = x
        self.y = y
        self.charge = charge

    def calc(self, size: int, world: list):
        for point in enumerate(world):
            x = point[0] % size
            y = int(point[0] / size)

            if x == self.x and y == self.y:
                world[point[0]] = [0, [0, 0]]
                continue

            vec_x = x - self.x
            vec_y = y - self.y

            vec_norm = sqrt(vec_x**2 + vec_y**2)

            force = (c_coulomb * self.charge) / vec_norm
            uni_mult = force**2 / vec_norm
            vec_x *= uni_mult
            vec_y *= uni_mult

            og_force = world[point[0]][0]
            world[point[0]] = [force + og_force]  # , [vec_x, vec_y]]


class World:
    def __init__(self, size: int):
        self.size = size
        self.world = []
        self.parts = []

        _base = [0, [0, 0]]

        for i in range(size**2):
            self.world.append(_base)

    def add_part(self, part: Particle):
        self.parts.append(part)

    def calc(self):
        for part in self.parts:
            part.calc(self.size, self.world)

    def get_world(self):
        return self.world

    def get_np(self):
        world_np = np.zeros((self.size, self.size))
        for y in range(self.size):
            for x in range(self.size):
                pos = x + self.size * y
                world_np[x, y] = self.world[pos][0]
        return world_np
