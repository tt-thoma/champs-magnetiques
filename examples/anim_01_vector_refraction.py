"""
Animation 1 VECTORIELLE : Refraction a travers un dielectrique (interface air-verre)
Montre les VECTEURS du champ magnetique se propageant et refractant.
"""

import subprocess

import matplotlib.pyplot as plt
import numpy as np

from champs_v4.fdtd_yee_3d import Yee3D
from champs_v4.visualization.vector_field_viz import VectorFieldVisualizer

from . import results_dir


def main():
    print("=" * 70)
    print(" " * 10 + "Animation 1 : REFRACTION - Vecteurs du champ")
    print("=" * 70)
    print()

    # Grille 2D
    nx, ny, nz = 200, 200, 1
    dx = 0.5e-3  # 0.5 mm
    c0 = 3e8
    dt = 0.45 * dx / (c0 * np.sqrt(2))

    print(f"Configuration :")
    print(f"  Grille : {nx}x{ny}x{nz}")
    print(f"  Resolution : dx = {dx * 1e3:.2f} mm")
    print(f"  Pas temps : dt = {dt * 1e12:.2f} ps")
    print()

    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=15)

    # Interface air/verre verticale
    sim.epsilon_r[nx // 2 :, :, :] = 2.25  # Verre (n=1.5)

    print(f"Materiaux :")
    print(f"  Gauche (x < {nx // 2}) : AIR (n = 1.0)")
    print(f"  Droite (x > {nx // 2}) : VERRE (n = 1.5)")
    print()

    # Source : impulsion gaussienne
    source_x = 30
    source_y = ny // 2
    freq = 8e9  # 8 GHz
    omega = 2 * np.pi * freq
    wavelength = c0 / freq

    print(f"Source :")
    print(f"  Position : x={source_x}, y={source_y}")
    print(f"  Type : Impulsion gaussienne")
    print(f"  Frequence : {freq / 1e9:.1f} GHz")
    print(f"  Lambda(air) : {wavelength * 1e3:.1f} mm")
    print(f"  Lambda(verre) : {wavelength * 1e3 / 1.5:.1f} mm")
    print()

    # Parametres impulsion
    t0 = 80 * dt
    spread = 25 * dt

    # Animation
    nsteps = 800
    frame_interval = 4

    print(f"Animation : {nsteps} pas, {nsteps // frame_interval} frames")
    print()

    # Dossier sortie
    out_dir = results_dir / "anim_01_vectors"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("Simulation et generation des frames...")
    frame_count = 0

    for n in range(nsteps):
        # Source impulsionnelle
        t = n * dt
        pulse = np.exp(-(((t - t0) / spread) ** 2)) * np.sin(omega * t)
        sim.Ez[source_x, source_y, 0] += pulse

        sim.step()

        # Generer frame
        if n % frame_interval == 0:
            if frame_count % 25 == 0:
                print(
                    f"  Frame {frame_count}/{nsteps // frame_interval} (pas {n}/{nsteps})"
                )

            # Creer visualiseur avec donnees actuelles
            viz = VectorFieldVisualizer(sim, field="auto", z_index=0)

            fig, axes = plt.subplots(1, 2, figsize=(16, 7))

            # Vue 1 : Vecteurs normalises
            viz.plot_normalized(
                axes[0], step=6, arrow_scale=3.5, show_magnitude_bg=True, cmap="turbo"
            )
            axes[0].axvline(
                nx // 2,
                color="white",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="Interface",
            )
            axes[0].legend(loc="upper right", fontsize=9)
            axes[0].set_title("Vecteurs normalises (directions)", fontsize=11)

            # Vue 2 : Streamlines
            viz.plot_streamlines(axes[1], density=0.9, color_by_magnitude=True)
            axes[1].axvline(
                nx // 2,
                color="cyan",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="Interface",
            )
            axes[1].legend(loc="upper right", fontsize=9)
            axes[1].set_title("Lignes de champ", fontsize=11)

            fig.suptitle(
                f"Refraction air/verre - t = {t * 1e12:.1f} ps",
                fontsize=13,
                fontweight="bold",
            )
            plt.tight_layout()

            frame_path = frames_dir / f"frame_{frame_count:04d}.png"
            plt.savefig(frame_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            frame_count += 1

    print()
    print(f"Total frames generees : {frame_count}")
    print(f"Dossier : {frames_dir}")
    print()

    # Generer video avec ffmpeg
    video_path = out_dir / "refraction_vectors.mp4"
    print(f"Generation video : {video_path.name}")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "30",
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(video_path),
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"Video creee : {video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Erreur ffmpeg : {e}")
        print("Frames disponibles mais video non creee")
    except FileNotFoundError:
        print("ffmpeg non trouve. Frames disponibles dans :", frames_dir)

    print()
    print("=" * 70)
    print("RESULTAT :")
    print("  - Onde se propage de gauche a droite")
    print("  - Refraction a l'interface air/verre")
    print("  - Changement de direction (loi de Snell)")
    print("  - Reflexion partielle visible")
    print("  - Vecteurs H tournent dans le plan")
    print("=" * 70)


if __name__ == "__main__":
    main()
