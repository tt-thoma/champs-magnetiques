"""
Animation 4 : Cavité diélectrique résonante
Montre la résonance dans une cavité entourée de diélectrique.
"""

import subprocess

import matplotlib.pyplot as plt
import numpy as np

from champs_v4.fdtd_yee_3d import Yee3D

from . import results_dir


def main():
    print("=" * 60)
    print("Animation 4 : Cavite dielectrique resonante")
    print("=" * 60)

    # Grille 2D
    nx, ny, nz = 160, 160, 1
    dx = 1.0e-3  # 1 mm
    c0 = 3e8
    dt = 0.45 * dx / (c0 * np.sqrt(2))

    print(f"Grille: {nx}x{ny}x{nz}, dx={dx * 1e3:.2f}mm")

    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=10)

    # Creer cavite rectangulaire avec dielectrique
    epsilon_r = np.ones((nx, ny, nz))
    sigma = np.zeros((nx, ny, nz))

    # Cavite : region centrale avec epsilon_r eleve (ceramique, epsilon_r environ 10)
    cavity_x = (50, 110)
    cavity_y = (50, 110)

    epsilon_r[cavity_x[0] : cavity_x[1], cavity_y[0] : cavity_y[1], :] = 10.0

    # Murs dielectriques (epsilon_r encore plus eleve pour confinement)
    wall_thickness = 5
    # Murs verticaux
    epsilon_r[
        cavity_x[0] : cavity_x[0] + wall_thickness, cavity_y[0] : cavity_y[1], :
    ] = 20.0
    epsilon_r[
        cavity_x[1] - wall_thickness : cavity_x[1], cavity_y[0] : cavity_y[1], :
    ] = 20.0
    # Murs horizontaux
    epsilon_r[
        cavity_x[0] : cavity_x[1], cavity_y[0] : cavity_y[0] + wall_thickness, :
    ] = 20.0
    epsilon_r[
        cavity_x[0] : cavity_x[1], cavity_y[1] - wall_thickness : cavity_y[1], :
    ] = 20.0

    sim.set_materials(epsilon_r, sigma)

    print(f"Cavite dielectrique :")
    print(f"  - Interieur : epsilon_r = 10 (ceramique)")
    print(f"  - Murs : epsilon_r = 20 (confinement)")
    print(
        f"  - Dimensions : {cavity_x[1] - cavity_x[0]} x {cavity_y[1] - cavity_y[0]} cellules"
    )

    # Source au centre de la cavité : impulsion courte
    source_pos = ((cavity_x[0] + cavity_x[1]) // 2, (cavity_y[0] + cavity_y[1]) // 2, 0)

    # Fréquence proche d'un mode de résonance
    cavity_size = (cavity_x[1] - cavity_x[0]) * dx
    n_dielectric = np.sqrt(10.0)
    # Mode fondamental : f ≈ c / (2 * L * n)
    f_resonance = c0 / (2 * cavity_size * n_dielectric)
    freq = f_resonance * 0.95  # Proche de la résonance
    omega = 2 * np.pi * freq

    print(f"Frequence de resonance estimee : {f_resonance / 1e9:.2f} GHz")
    print(f"Frequence source : {freq / 1e9:.2f} GHz")
    print(f"Type : Impulsion gaussienne courte")

    nsteps = 1500
    frame_interval = 8

    out_dir = results_dir / "anim_04_cavity"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Impulsion courte pour exciter les modes
    t0 = 60 * dt
    spread = 20 * dt

    print(f"Simulation de {nsteps} pas (observation de la résonance)...")

    frame_count = 0
    for n in range(nsteps):
        t = n * dt
        pulse = np.exp(-(((t - t0) / spread) ** 2)) * np.sin(omega * t)
        source_value = 0.2 * pulse

        sim.Ez[source_pos] += source_value
        sim.step()

        if n % frame_interval == 0:
            Ez_slice = sim.Ez[:, :, 0]
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(
                Ez_slice.T,
                origin="lower",
                cmap="coolwarm",
                vmin=-0.1,
                vmax=0.1,
                extent=[0, nx, 0, ny],
            )
            rect = plt.Rectangle(
                (cavity_x[0], cavity_y[0]),
                cavity_x[1] - cavity_x[0],
                cavity_y[1] - cavity_y[0],
                fill=False,
                edgecolor="yellow",
                linewidth=2,
                label="Cavité",
            )
            ax.add_patch(rect)
            ax.set_title(f"Résonance cavité diélectrique - Pas {n}/{nsteps}")
            ax.legend()
            plt.colorbar(im, ax=ax, label="Ez (V/m)")
            plt.tight_layout()
            plt.savefig(frames_dir / f"frame_{frame_count:04d}.png", dpi=100)
            plt.close(fig)
            frame_count += 1

        if n % 250 == 0:
            print(f"  Pas {n}/{nsteps} - {frame_count} frames")

    mp4_path = out_dir / "cavity_resonance.mp4"
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
    print("  Phenomene : Modes de resonance dans cavite dielectrique")


if __name__ == "__main__":
    main()
