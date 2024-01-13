# -*- encoding: utf-8 -*-
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import time
c_coulomb = 8.854 * (10**-12)
charge_unit = 10 * 1.602 * 10**-19

unit_time = 0.01
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
         for point, _ in enumerate(world):
            x = point % size
            y = point // size
        
            
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
       
       # Accélération
        accel_x = (world[id, 1] *self.charge)/ self.mass
        accel_y = (world[id, 2] *self.charge)/ self.mass
        print(world[id,2])
        # Mise à jour de la vitesse
        self.vx += accel_x * unit_time
        self.vy += accel_y * unit_time
        print(self.vy)
        # Nouvelle position
        self.x += self.vx * unit_time
        self.y += self.vy * unit_time
        
        if self.x <= 0:
            self.x = 1
  
        elif self.x >= size:  
            self.x = size - 2
 
        if self.y <= 0:
            self.y = 1
     
        elif self.y >= size:  
            self.y = size - 2  
      

class World:
    def __init__(self, size: int) -> None:
        self.size: int = size
        # Force, vecX, vecY
        self.world = np.zeros((size**2, 3))
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
            img[-y, x] = 1

        return img
    
    def animate(self, i):
        self.calc()
        self.img.set_array(self.get_pos())
        return [self.img]

    def create_animation(self):
        self.fig, self.ax = plt.subplots()
        self.ax.axis('on')
        self.img = self.ax.imshow(self.get_pos(), animated=True)
        ani = FuncAnimation(self.fig, self.animate, frames=1, interval=100, blit=True)
        
        plt.show()

world = World(8)
world.add_part(Particle(6,4,-100*charge_unit))
world.add_part(Particle(6,1,100*charge_unit))
world.create_animation()