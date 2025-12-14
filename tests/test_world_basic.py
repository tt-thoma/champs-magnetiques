import numpy as np
from world import World
from corps import Particle
import constants as const


def run_basic_test():
    # small world for quick test
    size = 0.2
    cell_size = 0.05
    dt = 0.01
    w = World(size, cell_size, dt, I=1.0, U=1.0, duree_simulation=0.1)

    # add a single particle near center
    center = size / 2
    p = Particle(center, center, center, const.charge_electron, const.masse_electron)
    w.add_part(p)

    # run a few steps
    for _ in range(3):
        w.step_FDTD()

    # basic checks: fields are finite and have correct shapes
    assert w.field_E.shape == w.field_B.shape
    assert np.isfinite(w.field_E).all(), "field_E contains inf/nan"
    assert np.isfinite(w.field_B).all(), "field_B contains inf/nan"
    print("Basic test passed: fields finite and shapes OK")


if __name__ == '__main__':
    run_basic_test()
