"""
Animation 5 VECTORIELLE : Interference dans structure multicouche
Montre les VECTEURS formant des interferences dans des couches multiples.
"""

import subprocess

import matplotlib.pyplot as plt
import numpy as np

from champs_v4.fdtd_yee_3d import Yee3D
from champs_v4.visualization.vector_field_viz import VectorFieldVisualizer

from . import results_dir


def main():
    print("=" * 70)
    print(" " * 8 + "Animation 5 : MULTICOUCHE - Vecteurs du champ")
    print("=" * 70)
    print()

    # Grille 2D
    nx, ny, nz = 240, 160, 1
    dx = 0.5e-3  # 0.5 mm
    c0 = 3e8
    dt = 0.45 * dx / (c0 * np.sqrt(2))

    print(f"Configuration :")
    print(f"  Grille : {nx}x{ny}x{nz}")
    print(f"  Resolution : dx = {dx * 1e3:.2f} mm")
    print()

    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=15)

    # Structure multicouche (3 couches alternees)
    layer_thickness = 30
    x_start = nx // 4

    # Couche 1 : air (par defaut epsilon_r = 1)
    x1 = x_start
    x2 = x1 + layer_thickness

    # Couche 2 : dielectrique 1 (verre)
    x3 = x2 + layer_thickness
    sim.epsilon_r[x2:x3, :, :] = 2.25  # n = 1.5

    # Couche 3 : dielectrique 2 (plastique)
    x4 = x3 + layer_thickness
    sim.epsilon_r[x3:x4, :, :] = 3.0  # n = 1.73

    # Couche 4 : retour air
    x5 = x4 + layer_thickness

    print(f"Structure multicouche (4 couches) :")
    print(f"  Couche 1 (x={x1}-{x2}) : AIR (n = 1.0)")
    print(f"  Couche 2 (x={x2}-{x3}) : VERRE (n = 1.5)")
    print(f"  Couche 3 (x={x3}-{x4}) : PLASTIQUE (n = 1.73)")
    print(f"  Couche 4 (x>{x4}) : AIR (n = 1.0)")
    print(f"  Epaisseur chaque couche : {layer_thickness} cellules")
    print()

    # Source
    source_x = 20
    source_y = ny // 2
    freq = 10e9  # 10 GHz
    omega = 2 * np.pi * freq
    wavelength = c0 / freq

    print(f"Source :")
    print(f"  Position : x={source_x}, y={source_y}")
    print(f"  Type : Impulsion gaussienne")
    print(f"  Frequence : {freq / 1e9:.1f} GHz")
    print(f"  Lambda : {wavelength * 1e3:.1f} mm")
    print()

    # Parametres impulsion
    t0 = 80 * dt
    spread = 25 * dt

    # Animation
    nsteps = 1000
    frame_interval = 5

    print(f"Animation : {nsteps} pas, {nsteps // frame_interval} frames")
    print()

    # Dossier sortie
    out_dir = results_dir / "anim_05_vectors"
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
            if frame_count % 20 == 0:
                print(
                    f"  Frame {frame_count}/{nsteps // frame_interval} (pas {n}/{nsteps})"
                )

            # Creer visualiseur avec donnees actuelles
            viz = VectorFieldVisualizer(sim, field="auto", z_index=0)

            fig, axes = plt.subplots(1, 2, figsize=(18, 7))

            # Vue 1 : Vecteurs normalises
            viz.plot_normalized(
                axes[0], step=7, arrow_scale=3.5, show_magnitude_bg=True, cmap="jet"
            )
            # Interfaces verticales
            axes[0].axvline(x2, color="white", linestyle="--", linewidth=1.5, alpha=0.6)
            axes[0].axvline(x3, color="white", linestyle="--", linewidth=1.5, alpha=0.6)
            axes[0].axvline(x4, color="white", linestyle="--", linewidth=1.5, alpha=0.6)
            # Labels couches
            axes[0].text(
                x1 + layer_thickness // 2,
                ny - 8,
                "AIR",
                color="white",
                fontsize=9,
                ha="center",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
            )
            axes[0].text(
                x2 + layer_thickness // 2,
                ny - 8,
                "VERRE",
                color="cyan",
                fontsize=9,
                ha="center",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
            )
            axes[0].text(
                x3 + layer_thickness // 2,
                ny - 8,
                "PLASTIQUE",
                color="yellow",
                fontsize=9,
                ha="center",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
            )
            axes[0].set_title("Vecteurs normalises", fontsize=11)

            # Vue 2 : Hybride
            viz.plot_hybrid(axes[1], streamline_density=0.8, quiver_step=14)
            axes[1].axvline(x2, color="cyan", linestyle="--", linewidth=1.5, alpha=0.6)
            axes[1].axvline(
                x3, color="yellow", linestyle="--", linewidth=1.5, alpha=0.6
            )
            axes[1].axvline(x4, color="white", linestyle="--", linewidth=1.5, alpha=0.6)
            axes[1].set_title("Vue hybride", fontsize=11)

            fig.suptitle(
                f"Interferences multicouches - t = {t * 1e12:.1f} ps",
                fontsize=13,
                fontweight="bold",
            )
            plt.tight_layout()

            frame_path = frames_dir / f"frame_{frame_count:04d}.png"
            plt.savefig(frame_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            frame_count += 1

    print()
    print(f"Total frames : {frame_count}")
    print()

    # Video
    video_path = out_dir / "multicouche_vectors.mp4"
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
        print(f"Video : {video_path}")
    except:
        print(f"Frames dans : {frames_dir}")

    print()
    print("=" * 70)
    print("RESULTAT :")
    print("  - Onde traverse les couches successives")
    print("  - Reflexions MULTIPLES aux interfaces")
    print("  - Interferences constructives/destructives")
    print("  - Patterns complexes de transmission")
    print("  - Vecteurs montrent directions variables")
    print("=" * 70)


if __name__ == "__main__":
    main()
