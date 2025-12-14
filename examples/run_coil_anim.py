import os

import datetime
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from champs_v4.fdtd_yee_3d import Yee3D

from . import base_dir as base

"""
Animate a coarse helical solenoid and its magnetic field (H) using Yee3D.
This script:
- builds a helical coil by placing short current segments (Jx, Jy) along a helix
- runs the FDTD solver for a modest number of steps
- samples H field on a coarse 3D grid, plots 3D quiver with the coil geometry,
  and saves frames to `results/frames_coil/`
- assembles frames to `results/coil_B_3d.mp4` using ffmpeg (if available)

Notes:
- This is a coarse, demonstrative model (not high-fidelity electromagnetics).
- Adjust `nx,ny,nz` and `nsteps` if you want higher resolution or longer runs.
"""


def add_helical_coil(sim: Yee3D, center, radius, z0, z1, turns, current, segments_per_turn=120):
    """Add helical coil by distributing tangential current segments into Jx and Jy arrays."""
    cx, cy = int(center[0]), int(center[1])
    pitch = (z1 - z0) / max(1.0, turns)
    total_samples = int(turns * segments_per_turn)
    # param s from 0..turns*2pi
    ss = np.linspace(0, 2 * np.pi * turns, total_samples, endpoint=False)
    for s in ss:
        x = cx + radius * np.cos(s)
        y = cy + radius * np.sin(s)
        z = z0 + (s / (2 * np.pi)) * pitch
        # tangent vector (dx/ds, dy/ds, dz/ds) -> (-r sin, r cos, pitch/(2pi))
        tx = -radius * np.sin(s)
        ty = radius * np.cos(s)
        tz = pitch / (2 * np.pi)
        # normalize tangent and scale by current; treat as current density along wire
        tnorm = np.sqrt(tx * tx + ty * ty + tz * tz) + 1e-12
        jx = (tx / tnorm) * current
        jy = (ty / tnorm) * current
        # place into nearest staggered grid indices
        ix = int(round(x))
        iy = int(round(y))
        iz = int(round(z))
        # Jx sits on Ex grid shape (nx, ny+1, nz+1)
        if 0 <= ix < sim.Jx.shape[0] and 0 <= iy < sim.Jx.shape[1] and 0 <= iz < sim.Jx.shape[2]:
            sim.Jx[ix, iy, iz] += jx
        # Jy sits on Ey grid
        if 0 <= ix < sim.Jy.shape[0] and 0 <= iy < sim.Jy.shape[1] and 0 <= iz < sim.Jy.shape[2]:
            sim.Jy[ix, iy, iz] += jy
    return


def main():
    out_dir = base / 'results'
    frames_dir = out_dir / 'frames_coil'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # grid
    nx, ny, nz = 80, 80, 40
    dx = 1e-3
    c0 = 299792458.0
    dt = 0.45 * dx / (c0 * np.sqrt(3))

    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=8, pml_sigma_max=2.5)

    # Helical coil parameters
    center = (nx // 2, ny // 2)
    radius = 12
    z0, z1 = 10, 30
    turns = 6
    current_amp = 1.0
    # use fewer segments for speed in this demo
    add_helical_coil(sim, center, radius, z0, z1, turns, current_amp, segments_per_turn=120)

    # Time-varying drive: simple sinusoidal multiplier applied to all J arrays each step
    freq = 5e7  # Hz (rough choice); you can change
    omega = 2 * np.pi * freq

    nsteps = 120
    frame_interval = 3
    frame_idx = 0

    # sampling grid for quiver (downsample for clarity)
    sx = slice(0, nx, 8)
    sy = slice(0, ny, 8)
    sz = slice(0, nz, 4)
    xs = np.arange(0, nx, 8)
    ys = np.arange(0, ny, 8)
    zs = np.arange(0, nz, 4)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    Jx_base = sim.Jx.copy()
    Jy_base = sim.Jy.copy()
    Jz_base = sim.Jz.copy()

    for n in range(nsteps):
        # modulate current (global multiplier)
        drive = 0.5 * (1.0 + np.sin(omega * n * dt))  # between 0 and 1
        # scale J arrays (we added steady J earlier; here we apply time multiplier)
        # For simplicity, scale whole arrays in-place by drive factor relative to baseline 1.0
        # (store baseline once on first iteration)
        sim.Jx[:] = Jx_base * drive
        sim.Jy[:] = Jy_base * drive
        sim.Jz[:] = Jz_base * drive

        sim.step()

        if (n % frame_interval) == 0:
            # sample H field on coarse grid
            Hx = sim.Hx[sx, sy, sz]
            Hy = sim.Hy[sx, sy, sz]
            Hz = sim.Hz[sx, sy, sz]
            # prepare vectors for plotting
            U = Hx.flatten()
            V = Hy.flatten()
            W = Hz.flatten()
            Xf = X.flatten()
            Yf = Y.flatten()
            Zf = Z.flatten()
            # create 3D quiver plot
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection='3d')
            # plot coil wire positions for context (approx helix points)
            svals = np.linspace(0, 2 * np.pi * turns, 600)
            hx = center[0] + radius * np.cos(svals)
            hy = center[1] + radius * np.sin(svals)
            hz = z0 + (svals / (2 * np.pi)) * ((z1 - z0) / turns)
            ax.plot(hx, hy, hz, color='red', linewidth=1.0)
            # normalize vectors for plotting
            mag = np.sqrt(U * U + V * V + W * W) + 1e-20
            # scale factor for arrows
            scale = 4
            ax.quiver(Xf, Yf, Zf, U / mag, V / mag, W / mag, length=scale, normalize=False, linewidth=0.5)
            ax.set_xlim(0, nx)
            ax.set_ylim(0, ny)
            ax.set_zlim(0, nz)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.view_init(elev=30, azim=(360.0 * frame_idx / (nsteps / frame_interval)))
            plt.title(f'Coil B-field (H) — frame {frame_idx}')
            plt.tight_layout()
            fname = frames_dir / f'frame_{frame_idx:04d}.png'
            plt.savefig(fname, dpi=120)
            plt.close(fig)
            frame_idx += 1
            print(f'Wrote frame {frame_idx}')

    # assemble into mp4 using ffmpeg if available
    out_mp4 = out_dir / 'coil_B_3d.mp4'
    try:
        # TODO: Fix
        cmd = ['ffmpeg', '-y', '-framerate', str(int(1.0 / (frame_interval * dt))), '-i', frames_dir / 'frame_%04d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_mp4]
        print('Running ffmpeg to assemble MP4 (this may take a while) ...')
        subprocess.run(cmd, check=True)
        print('MP4 assembled at', out_mp4)
    except Exception as e:
        print('ffmpeg failed or not found; frames are in', frames_dir)
        print('Exception:', e)

    # write metadata
    desc = out_dir / 'coil_animation_description.txt'
    with open(desc, 'w', encoding='utf-8') as f:
        f.write('Coil animation run\n')
        f.write('=================\n')
        f.write(f'Grid: nx={nx}, ny={ny}, nz={nz}, dx={dx} m\n')
        f.write(f'dt={dt} s, steps={nsteps}, frame_interval={frame_interval}\n')
        f.write(f'Coil: center={center}, radius={radius}, z0={z0}, z1={z1}, turns={turns}\n')
        f.write(f'Frames: {frame_idx} frames in {frames_dir}\n')
        f.write(f'Output MP4: {out_mp4}\n')
        f.write(f'timestamp={datetime.datetime.now().isoformat()}\n')

    print('Done. Description file:', desc)


if __name__ == '__main__':
    main()
