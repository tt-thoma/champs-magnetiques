import sys
sys.path.insert(0, r'c:\Espace de programation\champs-magnetiques\champs_v4')
from fdtd_yee_3d import Yee3D
import numpy as np
import math


def run_skin_depth_test():
    nx, ny, nz = 80, 10, 1
    dx = 1e-3
    dt = dx / (3e8 * np.sqrt(3)) * 0.5
    sim = Yee3D(nx, ny, nz, dx, dt)

    # create a conductor slab in the middle with sigma large
    sigma = np.zeros((nx, ny, nz))
    slab_start = 40
    sigma[slab_start:slab_start+10, :, :] = 1e7  # high conductivity
    sim.set_materials(np.ones((nx, ny, nz)), sigma)

    # source at i=5
    src_i = 5
    # Use a short Gaussian pulse injected repeatedly so the wave reaches the slab
    # compute number of steps required for wave to reach slab
    distance_m = (slab_start - src_i) * dx
    time_to_travel = distance_m / 3e8
    steps_travel = int(time_to_travel / dt) + 20

    # pulse parameters
    pulse_center = 20
    pulse_width = 6.0
    max_before = 0.0
    max_inside = 0.0

    total_steps = steps_travel + 80
    for n in range(total_steps):
        # gaussian pulse in time
        val = math.exp(-0.5 * ((n - pulse_center) / pulse_width) ** 2)
        sim.Ez[src_i, :, 0] = val
        sim.step()

        # after wave has passed source and travelled, sample
        if n > pulse_center:
            before_val = np.abs(sim.Ez[slab_start - 1, :, 0]).max()
            inside_val = np.abs(sim.Ez[slab_start + 2, :, 0]).max()
            if before_val > max_before:
                max_before = before_val
            if inside_val > max_inside:
                max_inside = inside_val

    print('before slab max Ez:', max_before, 'inside slab max Ez:', max_inside)
    # Expect significant attenuation inside conductor (orders of magnitude)
    assert max_before > 0.0, 'No wave detected before slab'
    assert max_inside < 0.5 * max_before, f'Field inside conductor not sufficiently attenuated ({max_inside} >= 0.5*{max_before})'
    print('Skin depth test passed')


if __name__ == '__main__':
    run_skin_depth_test()
