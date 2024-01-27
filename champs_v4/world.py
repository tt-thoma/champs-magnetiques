import numpy as np
from particle import Particle

class World:
    def __init__(self, size: int, dt: float) -> None:
        self.size: int = size
        self.dt: float = dt
        
        self.field_E = np.zeros((size, size, 2), float)
        self.field_B = np.zeros((size, size, 2), float)
        
        self.particles: list[Particle] = []

    def add_part(self, part: Particle) -> None:
        self.particles.append(part)