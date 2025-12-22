"""
Animation 1 : Réfraction à travers un diélectrique (interface air-verre)
Montre une onde plane traversant une interface entre deux milieux avec indices différents.
"""

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

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

    print("Materiaux configures :")
    print("  - Air (gauche) : epsilon_r = 1.0")
    print("  - Verre (droite) : epsilon_r = 2.25 (n = 1.5)")

    # Source : impulsion gaussienne
    source_x = 30
    freq = 8e9  # 8 GHz
    omega = 2 * np.pi * freq
    wavelength = c0 / freq

    print("Source : Impulsion gaussienne")
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

    Ez_slice = sim.Ez[:, :, 0]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        Ez_slice.T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
        extent=(0, nx, 0, ny),
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
    ax.set_title("Réfraction Air-Verre")
    ax.legend()
    plt.colorbar(im, ax=ax, label="Ez (V/m)")
    plt.tight_layout()

    rendered: list[int] = []

    # for n in range(nsteps):
    def update(frame: int) -> tuple[AxesImage,]:
        nonlocal rendered
        if frame in rendered:
            return (im,)
        else:
            rendered.append(frame)
        for _ in range(frame_interval):
            # Injection impulsion gaussienne
            t = frame * frame_interval * dt
            pulse = np.exp(-(((t - t0) / spread) ** 2)) * np.sin(omega * t)
            source_value = pulse

            # Injecter sur toute la colonne
            for j in range(ny):
                sim.Ez[source_x, j, 0] += source_value

            sim.step()

        Ez_slice = sim.Ez[:, :, 0]
        im.set_data(Ez_slice.T)

        if (frame * frame_interval) % 100 == 0:
            print(f"  Pas {frame * frame_interval}/{nsteps} - {frame} frames")

        return (im,)

    gif_path = out_dir / "refraction_animation.gif"
    ani = matplotlib.animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=nsteps // frame_interval,
        interval=(1 / 20) * 1000,  # 20 FPS -> milliseconds interval
        blit=True,
    )
    ani.save(gif_path, writer="ffmpeg")

    print("  Phénomène observé : Réfraction de Snell avec changement de vitesse")


if __name__ == "__main__":
    main()
