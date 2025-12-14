"""Visualization helpers: animate field lines (streamlines) on a 2D slice.

Functions:
- sample_field_slice(sim, field='B', axis='z', index=None): return (X,Y,U,V) at cell centers
- animate_slice(sim, field='B', axis='z', index=None, out_dir=..., nframes=..., frame_interval=...)

This module expects a `Yee3D` instance and extracts in-plane vector components by
averaging staggered arrays to cell centers. It uses matplotlib's `streamplot` to draw
field lines for E or H on a planar slice.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime


def _centered_H_plane(sim, k):
    """Return Hx_c, Hy_c on cell centers for z-index k (0..nz-1).
    Shapes returned: (nx, ny)
    """
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    # Hx shape: (nx+1, ny, nz) -> average along i to get (nx, ny)
    Hx_c = 0.5 * (sim.Hx[0:nx, :, k] + sim.Hx[1:nx + 1, :, k])
    # Hy shape: (nx, ny+1, nz) -> average along j
    Hy_c = 0.5 * (sim.Hy[:, 0:ny, k] + sim.Hy[:, 1:ny + 1, k])
    return Hx_c, Hy_c


def _centered_E_plane(sim, k):
    """Return Ex_c, Ey_c on cell centers for z-index k (0..nz-1).
    Shapes returned: (nx, ny)
    """
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    # Ex shape: (nx, ny+1, nz+1) -> average along j and pick k
    Ex_c = 0.5 * (sim.Ex[:, 0:ny, k] + sim.Ex[:, 1:ny + 1, k])
    # Ey shape: (nx+1, ny, nz+1) -> average along i and pick k
    Ey_c = 0.5 * (sim.Ey[0:nx, :, k] + sim.Ey[1:nx + 1, :, k])
    return Ex_c, Ey_c


def sample_field_slice(sim, field='B', axis='z', index=None):
    """Sample 2D vector field on a planar slice.
    - sim: Yee3D instance
    - field: 'B' or 'E' (magnetic H or electric E)
    - axis: 'z' only currently supported
    - index: slice index along axis (int). If None, selects middle slice.

    Returns: X, Y, U, V arrays on cell centers (shape (nx,ny)).
    """
    if axis != 'z':
        raise ValueError('Only axis="z" supported in this helper (can extend later).')
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    if index is None:
        k = nz // 2
    else:
        k = int(index)
        if k < 0 or k >= nz:
            raise IndexError('slice index out of bounds')
    if field.upper() in ('B', 'H'):
        U, V = _centered_H_plane(sim, k)
    elif field.upper() in ('E',):
        U, V = _centered_E_plane(sim, k)
    else:
        raise ValueError('field must be "B"/"H" or "E"')

    # create grid of centers
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y, indexing='ij')  # shapes (nx,ny)
    return X, Y, U, V


def animate_slice(sim, field='B', axis='z', index=None, out_dir=None, prefix='slice',
                  nsteps=120, frame_interval=3, sample_step=(1,1), cmap='viridis'):
    """Run the simulation for `nsteps` steps and create streamline frames for the specified slice.

    - sim: Yee3D instance (already configured)
    - field: 'B' or 'E'
    - index: z-slice index, None -> middle
    - out_dir: directory to save `frames` and `mp4`. If None, uses `./results` next to caller.
    - prefix: filename prefix
    - nsteps: number of time-steps to advance
    - frame_interval: save a frame every this many steps
    - sample_step: tuple (sx,sy) to downsample field for plotting readability (integers)
    - cmap: colormap for magnitude background

    Returns path to MP4 (if ffmpeg available) and description path.
    """
    base = out_dir or os.path.join(os.path.dirname(__file__), '..', 'results')
    frames_dir = os.path.join(base, f'frames_{prefix}')
    os.makedirs(frames_dir, exist_ok=True)

    nx, ny = sim.nx, sim.ny
    # prepare sampling grid
    sx, sy = sample_step
    xs = np.arange(0, nx, sx)
    ys = np.arange(0, ny, sy)
    Xs, Ys = np.meshgrid(xs, ys, indexing='ij')

    frame_idx = 0
    # run and save frames
    for n in range(nsteps):
        sim.step()
        if (n % frame_interval) != 0:
            continue
        X, Y, U_full, V_full = sample_field_slice(sim, field=field, axis=axis, index=index)
        # downsample for plotting
        U = U_full[0:nx:sx, 0:ny:sy]
        V = V_full[0:nx:sx, 0:ny:sy]
        Xp = Xs
        Yp = Ys
        mag = np.sqrt(U * U + V * V)

        fig, ax = plt.subplots(figsize=(6,5))
        # background magnitude
        im = ax.imshow(np.transpose(mag), origin='lower', cmap=cmap,
                       extent=[0, nx, 0, ny], aspect='equal')
        # streamlines require arrays on meshgrid Y,X order -> use transpose
        # Matplotlib streamplot expects shape (ny, nx) with x and y increasing arrays
        try:
            ax.streamplot(np.arange(nx), np.arange(ny), U.T, V.T, density=1.5, color='k', linewidth=0.6)
        except Exception:
            # fallback: quiver
            ax.quiver(Xp, Yp, U, V, pivot='mid', color='k', scale=50)
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_title(f'{field}-field lines (slice {index if index is not None else sim.nz//2}) step {n}')
        plt.colorbar(im, ax=ax, label='|field| (a.u.)')
        fname = os.path.join(frames_dir, f'{prefix}_frame_{frame_idx:04d}.png')
        plt.tight_layout()
        plt.savefig(fname, dpi=120)
        plt.close(fig)
        frame_idx += 1

    # assemble mp4 with ffmpeg if available
    mp4_path = os.path.join(base, f'{prefix}_slice_{field}.mp4')
    try:
        import subprocess
        cmd = ['ffmpeg', '-y', '-framerate', str(int(1.0 / (frame_interval * sim.dt))), '-i', os.path.join(frames_dir, f'{prefix}_frame_%04d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', mp4_path]
        subprocess.run(cmd, check=True)
    except Exception:
        mp4_path = None

    # write description
    desc = os.path.join(base, f'{prefix}_slice_{field}_description.txt')
    with open(desc, 'w', encoding='utf-8') as f:
        f.write('Slice field-line animation\n')
        f.write('========================\n')
        f.write(f'Grid: nx={sim.nx}, ny={sim.ny}, nz={sim.nz}, dx={getattr(sim, "dx", "?")}\n')
        f.write(f'dt={sim.dt} s, steps={nsteps}, frame_interval={frame_interval}\n')
        f.write(f'Field: {field}, axis={axis}, index={index}\n')
        f.write(f'Frames: {frame_idx} in {frames_dir}\n')
        f.write(f'Output MP4: {mp4_path}\n')
        f.write(f'timestamp={datetime.datetime.now().isoformat()}\n')

    return mp4_path, desc
