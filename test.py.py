import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

c_coulomb = 8.854 * (10**-12)
charge_unit = 10 * 1.602 * (10**-19)
unit_time = 0.01

class Particle:
    def __init__(self, x, y, charge, vx=0.0, vy=0.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.charge = charge
        self.old_x = x - vx * unit_time  # Initialiser old_x et old_y pour la méthode de Verlet
        self.old_y = y - vy * unit_time

        if charge < 0:
            self.mass = 9.11 * (10**-27)
        elif charge > 0:
            self.mass = 1.673 * (10**-27)
        else:
            self.mass = 1.675 * (10**-27)
        self.mass *= abs(charge)

    def calc(self, world, size):
        for point, _ in enumerate(world):
            x = round(point % size)
            y = round(point // size)
            self_x = round(self.x)
            self_y = round(self.y)

            if x == self.x and y == self.y:
                continue
            else:
                vec_x = x - self.x
                vec_y = y - self.y
                vec_norm = sqrt(vec_x**2 + vec_y**2)
                force = abs((c_coulomb * self.charge) / vec_norm**2)
                uni_mult = force / vec_norm
                vec_x *= uni_mult
                vec_y *= uni_mult
                og_force = world[point,0]
                world[point] = force + og_force ,  vec_x , vec_y

    def calc_next(self, world, size):
        id = round((self.x) + size * (self.y) )
        accel_x = (world[id, 1] *self.charge)/ self.mass
        accel_y = (world[id, 2] *self.charge)/ self.mass
        self.vx += accel_x * unit_time
        self.vy += accel_y * unit_time
        self.x += self.vx * unit_time
        self.y += self.vy * unit_time

    def collide(self, other):
        if sqrt((self.x - other.x)**2 + (self.y - other.y)**2) < 1:
            v1 = sqrt(self.vx**2 + self.vy**2)
            v2 = sqrt(other.vx**2 + other.vy**2)
            m1 = self.mass
            m2 = other.mass
            new_vx1 = ((m1 - m2) * self.vx + 2 * m2 * other.vx) / (m1 + m2)
            new_vy1 = ((m1 - m2) * self.vy + 2 * m2 * other.vy) / (m1 + m2)
            new_vx2 = ((m2 - m1) * other.vx + 2 * m1 * self.vx) / (m1 + m2)
            new_vy2 = ((m2 - m1) * other.vy + 2 * m1 * self.vy) / (m1 + m2)
            self.vx, self.vy = new_vx1, new_vy1
            other.vx, other.vy = new_vx2, new_vy2

    def verlet(self, dt):
        new_x = 2*self.x - self.old_x + self.ax*dt**2
        new_y = 2*self.y - self.old_y + self.ay*dt**2
        self.old_x, self.old_y = self.x, self.y
        self.x, self.y = new_x, new_y

class World:
    def __init__(self, size):
        self.size = size
        self.world = np.zeros((size**2, 3))
        self.particles = []

    def add_part(self, part):
        self.particles.append(part)

    def calc(self):
        for part in self.particles:
            part.calc(self.world, self.size)
        for i in range(len(self.particles)):
            for j in range(i+1, len(self.particles)):
                self.particles[i].collide(self.particles[j])
        for part in self.particles:
            part.verlet(unit_time)
            part.calc_next(self.world, self.size)

    def get_pos(self):
        img = np.zeros((self.size, self.size))
        for part in self.particles:
            x,y = round(part.x), round(part.y)
            if x > self.size or y > self.size:
                continue
            if x < 0 or y < 0:
                continue
            img[-y, x] = 10
        return img

    def animate(self, i):
        self.calc()
        self.img.set_array(self.get_pos())
        return [self.img]

    def create_animation(self):
        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')
        self.img = self.ax.imshow(self.get_pos(), animated=True)
        ani = FuncAnimation(self.fig, self.animate, frames=10000000, interval=1000, blit=True)
        plt.show()

world = World(100)
world.add_part(Particle(40,40,1*charge_unit))
world.add_part(Particle(60,60,1*charge_unit))
world.create_animation()
