"""
Animation 1 : Réfraction à travers un diélectrique (interface air-verre)
Montre une onde plane traversant une interface entre deux milieux avec indices différents.
"""

import subprocess

import matplotlib.pyplot as plt
import numpy as np

from champs_v4.fdtd_yee_3d import Yee3D

from . import results_dir


def main():
    print("=" * 60)
    print("Animation 1 : Refraction a travers un dielectrique")
    print("=" * 60)

    # Grille 2D (nz=1 pour simulation 2D)
    nx, ny, nz = 200, 200, 1
    dx = 0.5e-3  # 0.5 mm
    c0 = 3e8
    dt = 0.45 * dx / (c0 * np.sqrt(2))  # CFL pour 2D

    print(f"Grille: {nx}x{ny}x{nz}, dx={dx * 1e3:.2f}mm, dt={dt * 1e12:.2f}ps")

    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=15)

    # Créer matériaux : interface verticale air/verre
    epsilon_r = np.ones((nx, ny, nz))

    # Verre (εr ≈ 2.25, n ≈ 1.5) dans la moitié droite
    epsilon_r[nx // 2 :, :, :] = 2.25

    # Pas de conductivité (milieux transparents)
    sigma = np.zeros((nx, ny, nz))

    sim.set_materials(epsilon_r, sigma)

    print(f"Materiaux configures :")
    print(f"  - Air (gauche) : epsilon_r = 1.0")
    print(f"  - Verre (droite) : epsilon_r = 2.25 (n = 1.5)")

    # Source : impulsion gaussienne
    source_x = 30
    freq = 8e9  # 8 GHz
    omega = 2 * np.pi * freq
    wavelength = c0 / freq

    print(f"Source : Impulsion gaussienne")
    print(f"Frequence : {freq / 1e9:.1f} GHz (lambda = {wavelength * 1e3:.2f} mm)")

    # Simulation avec génération de frames
    nsteps = 800
    frame_interval = 4

    print(f"Simulation : {nsteps} pas, {nsteps // frame_interval} frames")

    # Créer dossier pour frames
    out_dir = results_dir / "anim_01_dielectric"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Paramètres impulsion gaussienne
    t0 = 80 * dt
    spread = 25 * dt

    frame_count = 0
    for n in range(nsteps):
        # Injection impulsion gaussienne
        t = n * dt
        pulse = np.exp(-(((t - t0) / spread) ** 2)) * np.sin(omega * t)
        source_value = pulse

        # Injecter sur toute la colonne
        for j in range(ny):
            sim.Ez[source_x, j, 0] += source_value

        sim.step()

        # Sauvegarder frame
        if n % frame_interval == 0:
            Ez_slice = sim.Ez[:, :, 0]

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(
                Ez_slice.T,
                origin="lower",
                cmap="RdBu_r",
                vmin=-0.5,
                vmax=0.5,
                extent=[0, nx, 0, ny],
            )
            ax.axvline(
                x=nx // 2,
                color="yellow",
                linestyle="--",
                linewidth=2,
                alpha=0.5,
                label="Interface",
            )
            ax.set_xlabel("X (cellules)")
            ax.set_ylabel("Y (cellules)")
            ax.set_title(f"Réfraction Air-Verre - Pas {n}/{nsteps}")
            ax.legend()
            plt.colorbar(im, ax=ax, label="Ez (V/m)")
            plt.tight_layout()

            frame_path = frames_dir / f"frame_{frame_count:04d}.png"
            plt.savefig(frame_path, dpi=100)
            plt.close(fig)
            frame_count += 1

        if n % 100 == 0:
            print(f"  Pas {n}/{nsteps} - {frame_count} frames")

    print(f"Simulation terminée, {frame_count} frames générées")

    # Créer MP4 avec ffmpeg
    mp4_path = out_dir / "refraction_animation.mp4"
    try:
        cmd = [
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
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"\n✓ Animation MP4 créée : {mp4_path}")
    except subprocess.CalledProcessError:
        print(f"\n⚠ FFmpeg erreur, frames PNG disponibles dans : {frames_dir}")
    except FileNotFoundError:
        print(f"\n⚠ FFmpeg non trouvé, frames PNG disponibles dans : {frames_dir}")

    print("  Phénomène observé : Réfraction de Snell avec changement de vitesse")


if __name__ == "__main__":
    main()
