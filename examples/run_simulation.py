import numpy as np
import matplotlib.pyplot as plt

from champs_v4.fdtd_yee_3d import Yee3D

from . import base_dir


def main():
    # Small demo: plane wave pulse propagating in x, snapshot Ez
    out_dir = base_dir / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)

    nx, ny, nz = 80, 40, 1
    dx = 1e-3
    # speed of light
    c0 = 299792458.0
    # CFL safe dt
    dt = 0.5 * dx / (c0 * np.sqrt(3))

    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=10, pml_sigma_max=2.0)

    # set a small region with conductivity to show skin effect optionally
    sigma = np.zeros((nx, ny, nz), dtype=float)
    # place a slab at x ~ 50..nx-1
    sigma[50:60, :, :] = 1e6  # S/m (very conductive slab)
    sim.set_materials(np.ones((nx, ny, nz)), sigma)

    # Inject a Gaussian pulse at x=5 (in Ez component indices)
    pulse_pos = 5
    nsteps = 300
    for n in range(nsteps):
        # time-dependent source on Ez at plane pulse_pos
        t = n * dt
        # gaussian temporal pulse
        src = np.exp(-((t - 5e-9) ** 2) / (2 * (1e-9) ** 2))
        # set Ez at cell plane (use Ez stagger: indices [i,j,k])
        if 0 <= pulse_pos < sim.Ez.shape[0]:
            sim.Ez[pulse_pos, sim.Ez.shape[1] // 2, 0] += src * 1.0
        sim.step()

    # take Ez slice (x,y) by collapsing z
    Ez = sim.Ez[:, :, 0]

    # save image
    out_path = out_dir / 'ez_snapshot.png'
    plt.figure(figsize=(8, 4))
    plt.imshow(Ez.T, origin='lower', cmap='RdBu', aspect='auto')
    plt.colorbar(label='Ez (a.u.)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ez snapshot after {} steps'.format(nsteps))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    # write descriptive text file alongside snapshot
    desc_path = out_path.with_suffix(".txt")
    with open(desc_path, 'w', encoding='utf-8') as f:
        f.write('Simulation snapshot description\n')
        f.write('===============================\n')
        f.write(f'Grid: nx={nx}, ny={ny}, nz={nz}\n')
        f.write(f'dx={dx} m, dt={dt} s\n')
        f.write(f'PML: width={sim.pml_width}, sigma_max={sim.pml_sigma_max}\n')
        f.write(f'Conductive slab ranges (nonzero sigma):\n')
        nonzero = np.argwhere(sigma > 0)
        if nonzero.size == 0:
            f.write('  none\n')
        else:
            mins = nonzero.min(axis=0)
            maxs = nonzero.max(axis=0)
            f.write(f'  x: {mins[0]}..{maxs[0]}, y: {mins[1]}..{maxs[1]}, z: {mins[2]}..{maxs[2]}\n')
            f.write(f'  sigma example value: {sigma[nonzero[0][0], nonzero[0][1], nonzero[0][2]]}\n')
        f.write(f'source pulse position (Ez indices): x={pulse_pos}, y={sim.Ez.shape[1]//2}, z=0\n')
        f.write(f'nsteps={nsteps}\n')
        import datetime
        f.write(f'timestamp={datetime.datetime.now().isoformat()}\n')
    print('Saved Ez snapshot to', out_path)
    print('Saved description to', desc_path)


if __name__ == '__main__':
    main()
