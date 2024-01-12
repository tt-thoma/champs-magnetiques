# -*- encoding: utf-8 -*-
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import time
c_coulomb = 8.854 * (10**-12)
charge_unit = 10 * 1.602 * 10**-19

unit_time = 0.5
m2mm = 10**-12
mm2m = 10**12


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
        self.mass *= abs(charge)
        
    def calc(self, world, size):
        for point in enumerate(world):
            print( point)
            x = point[0] % size
            y = int(point[0] / size)
            
            if x == self.x and y == self.y:
                print(world[point[0],0])
                
                
                
            else:
                           
                vec_x = x - self.x
                vec_y = y - self.y

                vec_norm = sqrt(vec_x**2 + vec_y**2)

                force = abs((c_coulomb * self.charge) / vec_norm**2)
                uni_mult = force / vec_norm
                vec_x *= uni_mult
                vec_y *= uni_mult
                
                og_force = world[point[0],0]
                
                world[point[0]] = force + og_force ,  vec_x , vec_y
          
    def calc_next(self, world, size):
        id = (round(self.x) + size * round(self.y)) % size

       # Accélération
        accel_x = (world[id, 1] *self.charge)/ self.mass
        accel_y = (world[id, 2] *self.charge)/ self.mass

        # Mise à jour de la vitesse
        self.vx += accel_x * unit_time
        self.vy += accel_y * unit_time
    
        # Nouvelle position
        self.x += self.vx * unit_time
        self.y += self.vy * unit_time
        
        # Vérifier les limites du monde et ajuster la position et la vitesse si nécessaire
        if self.x < 0:
            self.x = 0
            self.vx = -self.vx 
            accel_x = -accel_x
        elif self.x > size:
            self.x = size
            self.vx = -self.vx
            accel_x = -accel_x
        if self.y < 0:
            self.y = 0
            self.vy = -self.vy
            accel_y = -accel_y
        elif self.y > size:
            self.y = size
            self.vy = -self.vy
            accel_y = -accel_y
        

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
        for part in self.particles:
            part.calc_next(self.world, self.size)



    def get_pos(self):
        img = np.zeros((self.size, self.size))
        for part in self.particles:
            x,y = round(part.x), round(part.y)
            if x > self.size or y > self.size:
                continue
            if x < 0 or y < 0:
                continue

            print(f"x: {x}, y: {y}")
            img[y, x] = 2

        return img
    
    def animate(self, i):
        self.calc()
        self.img.set_array(self.get_pos())
        return [self.img]

    def create_animation(self):
        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')
        self.img = self.ax.imshow(self.get_pos(), animated=True)
        ani = FuncAnimation(self.fig, self.animate, frames=10, interval=200000, blit=True)
        plt.show()

    def get_np(self):
        raise NotImplementedError


world = World(2)

world.add_part(Particle(1,1,1000*charge_unit))




world.create_animation()
