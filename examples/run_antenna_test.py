import os
import sys
import numpy as np

# Ensure repository root is on sys.path
_here = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_here, '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from fdtd_yee_3d import Yee3D
from visualization.animation_module import create_animation

def main():
    # Configuration pour un test rapide d'antenne
    base_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    out_dir = os.path.join(base_dir, '..', 'results', 'antenna_test')
    frames_dir = os.path.join(out_dir, 'frames')

    # Grille petite pour test rapide
    nx, ny, nz = 40, 40, 40
    dx = 3e-3  # 3 mm
    c0 = 3e8
    dt = 0.5 * dx / c0  # CFL conservateur

    print(f"Grille: {nx}x{ny}x{nz}, dx={dx*1e3:.1f}mm, dt={dt*1e12:.1f}ps")

    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=5)

    # Fréquence 1 GHz
    f = 1e9
    omega = 2 * np.pi * f

    print(f"Fréquence: {f/1e6} MHz")

    # Simulation courte: 10 périodes
    periods = 10
    t_total = periods / f
    nsteps = int(t_total / dt)
    frame_interval = max(1, nsteps // 20)  # 20 frames

    print(f"Simulation: {periods} périodes, {nsteps} pas, {nsteps//frame_interval} frames")

    # Créer l'animation 3D
    mp4_path = create_animation(
        frames_dir=frames_dir,
        output_file=os.path.join(out_dir, 'antenna_test_3d.mp4'),
        framerate=5,
        mode='3d',
        sim=sim,
        field='E',
        nsteps=nsteps,
        frame_interval=frame_interval,
        sx=2, sy=2, sz=2,  # Sous-échantillonnage réduit pour petite grille
        source_type='antenna',
        omega=omega,
        feed_pos=(nx//2, ny//2, nz//2),
        amplitude=0.01,
        envelope_type='gaussian',
        t0=t_total / 2,
        width=t_total / 20
    )

    if mp4_path:
        print(f"Animation test créée: {mp4_path}")
    else:
        print("Échec de création de l'animation.")

if __name__ == '__main__':
    main()