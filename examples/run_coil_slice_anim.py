import numpy as np

from champs_v4.fdtd_yee_3d import Yee3D
from champs_v4.visualization.field_slice_anim import animate_slice

from . import base_dir as base

"""
Example runner: construct coarse solenoid, run brief simulation, and create a 2D slice
animation of magnetic field lines using `animate_slice`.
"""

def add_solenoid_simple(sim, center, radius_cells, z0, z1, turns, current_per_turn, thickness=1):
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    ix0, iy0 = int(center[0]), int(center[1])
    z_indices = np.linspace(z0, z1, turns).astype(int)
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    dist = np.sqrt((x - ix0) ** 2 + (y - iy0) ** 2)
    for k in z_indices:
        mask = (dist >= (radius_cells - 0.5)) & (dist <= (radius_cells + thickness - 0.5))
        circumference = 2 * np.pi * max(radius_cells, 1)
        J_cell = current_per_turn / max(circumference, 1.0)
        for i in range(nx):
            for j in range(ny):
                if mask[i, j]:
                    if 0 <= i < sim.Jz.shape[0] and 0 <= j < sim.Jz.shape[1] and 0 <= k < sim.Jz.shape[2]:
                        sim.Jz[i, j, k] += J_cell


def main():
    results_dir = base / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # small grid for demo
    nx, ny, nz = 80, 80, 40
    dx = 1e-3
    c0 = 299792458.0
    dt = 0.45 * dx / (c0 * np.sqrt(3))
    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=8, pml_sigma_max=2.5)

    center = (nx // 2, ny // 2)
    radius_cells = 12
    z0, z1 = 10, 30
    turns = 20
    current_per_turn = 1.0
    add_solenoid_simple(sim, center, radius_cells, z0, z1, turns, current_per_turn, thickness=1)

    # run and create slice animation of magnetic field (B/H)
    mp4, desc = animate_slice(sim, field='B', axis='z', index=(z0 + z1)//2,
                              out_dir=results_dir, prefix='coil_slice', nsteps=120,
                              frame_interval=3, sample_step=(2,2))
    print('Animation MP4:', mp4)
    print('Description:', desc)

if __name__ == '__main__':
    main()
