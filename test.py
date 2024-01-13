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
        #self.mass *= abs(charge)
        
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
                if vec_norm != 0:
                    force = abs((c_coulomb * self.charge) / vec_norm**2)
                    repulsion = 1 / vec_norm**2 if vec_norm < 1 else 0
                    force += repulsion

                    uni_mult = force / vec_norm
                    vec_x *= uni_mult
                    vec_y *= uni_mult  
                else:
                    # Gérer le cas où vec_norm est zéro
                    force = 0
                    vec_x = 0
                    vec_y = 0
             
                
                og_force = world[point,0]      
                world[point] = force + og_force ,  vec_x , vec_y
    def calc_next(self, world, size, dt):
            id = round((self.x) + size * (self.y) )
            # Accélération
            accel_x = (world[id, 1] *self.charge)/ self.mass
            accel_y = (world[id, 2] *self.charge)/ self.mass
            # Mise à jour de la vitesse et de la position avec RK4
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

            self.vx += (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx) / 6
            self.vy += (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy) / 6
            self.x += (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
            self.y += (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
            print(f"accel_x = {accel_x}, accel_y = {accel_y}, vx = {self.vx}  vy = {self.vx}")
            if self.x <= 0:
                self.x = 0
            elif self.x >= size:  
                self.x = size - 1
            if self.y <= 0:
                self.y = 0
            elif self.y >= size:  
                self.y = size - 1 
class World:
    def __init__(self, size: int, dt: float) -> None:
        self.size: int = size
        self.dt :float = dt
        # Force, vecX, vecY
        self.world = np.zeros((size**2, 3))
        self.particles: list[Particle] = []
    def add_part(self, part: Particle) -> None:
        self.particles.append(part)
    def calc(self) -> None:
        for part in self.particles:
            part.calc(self.world, self.size)
        for part in self.particles:
            part.calc_next(self.world, self.size, self.dt)
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
        ani = FuncAnimation(self.fig, self.animate, frames=100, interval=0, blit=True)
        
        plt.show()

world = World(100,0.01)
world.add_part(Particle(60,40,-500000*charge_unit))
world.add_part(Particle(60,60,500000*charge_unit))
dt = 0.01   

print(f"condition initial : 2 particules 60,40,-100000*charge_unit et 60,60,100000*charge_unit dans un monde 100,0.01 avec dt = 0.01")
world.create_animation()