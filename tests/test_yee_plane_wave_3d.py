import sys
sys.path.insert(0, r'c:\Espace de programation\champs-magnetiques\champs_v4')
from fdtd_yee_3d import Yee3D, c0
import numpy as np
import math


def run_plane_wave_test():
    # small 3D domain with nz=1 (effectively 2D XY)
    nx, ny, nz = 50, 50, 1
    dx = 1e-3
    # CFL for Yee: dt <= dx/(c*sqrt(3)) -> choose safe dt
    dt = dx / (c0 * np.sqrt(3)) * 0.5
    sim = Yee3D(nx, ny, nz, dx, dt)

    # set a harmonic source on Ez at x=2 plane
    freq = 1e9
    omega = 2 * np.pi * freq
    tsteps = 60

    src_i = 2
    max_val = 0.0
    for n in range(tsteps):
        t = n * dt
        # inject Ez at a plane (j varies)
        val = math.sin(omega * t)
        # write into Ez indices: Ez shape (nx+1, ny+1, nz)
        sim.Ez[src_i, :, 0] = val
        sim.step()
        # sample Ez at a further plane
        sample = np.abs(sim.Ez[src_i + 10, :, 0]).max()
        if sample > max_val:
            max_val = sample

    assert max_val > 1e-6, 'Plane wave did not propagate (max sample too small)'
    print('Plane wave test: passed, max sampled Ez =', max_val)


if __name__ == '__main__':
    import math
    run_plane_wave_test()
