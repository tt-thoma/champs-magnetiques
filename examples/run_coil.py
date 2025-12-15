import numpy as np
import matplotlib.pyplot as plt

from champs_v4.fdtd_yee_3d import Yee3D

from . import base_dir

# Example: build a solenoid (bobine) as a stack of circular current loops
# This script approximates each turn by adding Jz on grid cells near the ring.


def add_solenoid(sim: Yee3D, center, radius_cells, z0, z1, turns, current_per_turn, thickness=1):
    """
    Add a solenoid to the simulation by setting Jz on cells approximating circular loops.
    - center: (ix,iy) center in cell indices
    - radius_cells: radius in cells
    - z0,z1: z index range (inclusive) where turns are placed
    - turns: number of turns (distributed between z0 and z1)
    - current_per_turn: amplitude (arbitrary units)
    - thickness: radial thickness in cells of each loop
    """
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    ix0, iy0 = int(center[0]), int(center[1])
    z_indices = np.linspace(z0, z1, turns).astype(int)
    # prepare grid of distances in xy plane
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    dist = np.sqrt((x - ix0) ** 2 + (y - iy0) ** 2)
    for k in z_indices:
        # ring mask where distance close to radius_cells (within thickness)
        mask = (dist >= (radius_cells - 0.5)) & (dist <= (radius_cells + thickness - 0.5))
        # distribute current density: add to Jz at (i,j,k)
        # current per cell is arbitrary scaling; divide by approximate circumference*cell_area
        circumference = 2 * np.pi * max(radius_cells, 1)
        cell_area = (sim.dx if hasattr(sim,'dx') else sim.dx if hasattr(sim,'dx') else 1.0)**2
        # Use simple normalization to keep amplitude reasonable
        J_cell = current_per_turn / max(circumference, 1.0)
        # Jz shape matches Ez grid (nx+1, ny+1, nz) in Yee3D - but sim.Jz was created same shape as Ez
        # we will safely clip indices when writing
        for i in range(nx):
            for j in range(ny):
                if mask[i, j]:
                    if 0 <= i < sim.Jz.shape[0] and 0 <= j < sim.Jz.shape[1] and 0 <= k < sim.Jz.shape[2]:
                        sim.Jz[i, j, k] += J_cell


def main():
    out_dir = base_dir / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)

    # grid and time
    nx, ny, nz = 80, 80, 40
    dx = 1e-3
    c0 = 299792458.0
    dt = 0.45 * dx / (c0 * np.sqrt(3))

    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=8, pml_sigma_max=2.5)

    # solenoid parameters (in grid cells)
    center = (nx // 2, ny // 2)
    radius_cells = 12
    z0, z1 = 10, 30
    turns = 20
    current_per_turn = 1.0

    add_solenoid(sim, center, radius_cells, z0, z1, turns, current_per_turn, thickness=1)

    # run for a number of timesteps so B field can develop
    nsteps = 400
    # optionally drive the coil (here we keep steady current by leaving J fixed)
    for n in range(nsteps):
        sim.step()
        if n % 50 == 0:
            print(f'step {n}/{nsteps}')

    # compute B magnitude (Hx,Hy,Hz) on a mid-slice (z mid)
    zm = (z0 + z1) // 2
    # H arrays shapes may be staggered; choose nearest indices and compute magnitude
    Hx = sim.Hx[:, :, zm]
    Hy = sim.Hy[:, :, zm]
    Hz = sim.Hz[:, :, zm]
    # ensure shapes consistent by clipping
    minx = min(Hx.shape[0], Hy.shape[0], Hz.shape[0])
    miny = min(Hx.shape[1], Hy.shape[1], Hz.shape[1])
    Hmag = np.sqrt(Hx[:minx, :miny] ** 2 + Hy[:minx, :miny] ** 2 + Hz[:minx, :miny] ** 2)

    # save snapshot
    out_png = out_dir / 'coil_B_mid_slice.png'
    plt.figure(figsize=(6,6))
    plt.imshow(Hmag.T, origin='lower', cmap='inferno')
    plt.colorbar(label='|H| (a.u.)')
    plt.title('Magnetic field magnitude (mid z slice)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    # write description
    desc_path = out_dir / 'coil_simulation_description.txt'
    import datetime
    with open(desc_path, 'w', encoding='utf-8') as f:
        f.write('Solenoid (bobine) simulation\n')
        f.write('==========================\n')
        f.write(f'Grid: nx={nx}, ny={ny}, nz={nz}, dx={dx} m\n')
        f.write(f'dt={dt} s, steps={nsteps}\n')
        f.write(f'PML: width={sim.pml_width}, sigma_max={sim.pml_sigma_max}\n')
        f.write('Solenoid parameters (grid cells):\n')
        f.write(f'  center={center}, radius={radius_cells}, z0={z0}, z1={z1}, turns={turns}\n')
        f.write(f'  current_per_turn={current_per_turn}\n')
        f.write(f'Output image: {out_png.name}\n')
        f.write(f'timestamp={datetime.datetime.now().isoformat()}\n')

    print('Saved B field snapshot to', out_png)
    print('Saved description to', desc_path)


if __name__ == '__main__':
    main()
