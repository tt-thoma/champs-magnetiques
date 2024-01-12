# -*- encoding: utf-8 -*-
from matplotlib import pyplot as plt
import champ_electrique_v2 as ce

print("Test file")
world = ce.World(100)
for _ in range(2):
    world.add_part(ce.Particle(40, 40, -4))
    world.add_part(ce.Particle(60, 60, 4))

for _ in range(10):
    world.calc()
    img = plt.imshow(world.get_pos())
    plt.show()
