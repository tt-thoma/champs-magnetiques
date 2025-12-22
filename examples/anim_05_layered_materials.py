"""
Animation 5 : Structure multicouche (réflexions multiples)
Montre les interférences dans un empilement de couches diélectriques.
"""

import subprocess

import matplotlib.pyplot as plt
import numpy as np

from champs_v4.fdtd_yee_3d import Yee3D

from . import results_dir


def main():
    print("=" * 60)
    print("Animation 5 : Structure multicouche avec interferences")
    print("=" * 60)

    # Grille 2D
    nx, ny, nz = 240, 200, 1
    dx = 0.4e-3  # 0.4 mm
    c0 = 3e8
    dt = 0.45 * dx / (c0 * np.sqrt(2))

    print(f"Grille: {nx}x{ny}x{nz}, dx={dx * 1e3:.2f}mm")

    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=15)

    # Creer structure multicouche verticale
    epsilon_r = np.ones((nx, ny, nz))
    sigma = np.zeros((nx, ny, nz))

    # Definir les couches (alternance de materiaux)
    layers = [
        (60, 80, 4.0, "Verre"),  # Couche 1
        (80, 100, 2.0, "Plastique"),  # Couche 2
        (100, 120, 9.0, "Ceramique"),  # Couche 3
        (120, 140, 2.5, "Teflon"),  # Couche 4
        (140, 160, 6.0, "Resine"),  # Couche 5
    ]

    print("\nCouches dielectriques :")
    for x_start, x_end, eps, name in layers:
        epsilon_r[x_start:x_end, :, :] = eps
        thickness = (x_end - x_start) * dx * 1e3
        print(
            f"  - {name:12s} : x=[{x_start:3d},{x_end:3d}], epsilon_r={eps:4.1f}, "
            f"epaisseur={thickness:.2f}mm"
        )

    sim.set_materials(epsilon_r, sigma)

    # Source : impulsion gaussienne
    source_x = 25
    freq = 10e9  # 10 GHz
    omega = 2 * np.pi * freq
    wavelength = c0 / freq

    print(f"\nSource : Impulsion gaussienne")
    print(f"Frequence : {freq / 1e9:.1f} GHz (lambda_0 = {wavelength * 1e3:.3f} mm)")

    nsteps = 1000
    frame_interval = 5

    out_dir = results_dir / "anim_05_multilayer"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Parametres impulsion gaussienne
    t0 = 80 * dt
    spread = 25 * dt

    print(f"\nSimulation de {nsteps} pas (réflexions multiples)...")

    frame_count = 0
    for n in range(nsteps):
        t = n * dt
        pulse = np.exp(-(((t - t0) / spread) ** 2)) * np.sin(omega * t)

        for j in range(ny):
            sim.Ez[source_x, j, 0] += pulse

        sim.step()

        if n % frame_interval == 0:
            Ez_slice = sim.Ez[:, :, 0]
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(
                Ez_slice.T,
                origin="lower",
                cmap="viridis",
                vmin=-0.3,
                vmax=0.3,
                extent=[0, nx, 0, ny],
            )

            # Marquer les couches
            for x_start, x_end, eps, name in layers:
                ax.axvline(
                    x=x_start, color="red", linestyle=":", linewidth=1, alpha=0.5
                )
                ax.axvline(x=x_end, color="red", linestyle=":", linewidth=1, alpha=0.5)

            ax.set_title(f"Interférences multicouche - Pas {n}/{nsteps}")
            ax.set_xlabel("X (cellules)")
            ax.set_ylabel("Y (cellules)")
            plt.colorbar(im, ax=ax, label="Ez (V/m)")
            plt.tight_layout()
            plt.savefig(frames_dir / f"frame_{frame_count:04d}.png", dpi=100)
            plt.close(fig)
            frame_count += 1

        if n % 200 == 0:
            print(f"  Pas {n}/{nsteps} - {frame_count} frames")

    mp4_path = out_dir / "multilayer.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "20",
                "-i",
                str(frames_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                str(mp4_path),
            ],
            check=True,
            capture_output=True,
        )
        print(f"\nOK Animation creee : {mp4_path}")
    except:
        print(f"\nATT Frames dans : {frames_dir}")

    print(f"OK Animation sauvegardee dans : {out_dir}")
    print("  Phenomene : Interferences constructives/destructives dans multicouche")
    print("  Application : Revetements antireflets, filtres optiques")


if __name__ == "__main__":
    main()
