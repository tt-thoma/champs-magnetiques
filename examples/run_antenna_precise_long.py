import numpy as np

from champs_v4.fdtd_yee_3d import Yee3D
from champs_v4.visualization.animation_module import create_animation

from . import base_dir

def main():
    # Configuration pour une antenne dipôle précise et longue
    out_dir = base_dir / 'results' / 'antenna_precise_long'
    frames_dir = out_dir / 'frames'

    # Grille haute résolution pour précision
    nx, ny, nz = 80, 80, 80
    dx = 3e-3  # 3 mm pour bonne résolution (lambda/dx ≈ 100)
    c0 = 3e8
    dt = 0.5 * dx / c0  # CFL plus conservateur pour stabilité

    print(f"Grille: {nx}x{ny}x{nz}, dx={dx*1e3:.1f}mm, dt={dt*1e12:.1f}ps")

    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=10)

    # Fréquence 1 GHz pour antenne
    f = 1e9
    omega = 2 * np.pi * f
    wavelength = c0 / f
    dipole_length = wavelength / 2

    print(f"Fréquence: {f/1e6} MHz, λ = {wavelength:.3f}m, longueur dipôle: {dipole_length:.3f}m")

    # Simulation longue: 200 périodes pour animation détaillée
    periods = 200
    t_total = periods / f
    nsteps = int(t_total / dt)
    frame_interval = max(1, nsteps // 100)  # ~100 frames

    print(f"Simulation: {periods} périodes, {nsteps} pas, {nsteps//frame_interval} frames")
    print(f"Temps estimé: ~{nsteps * 0.05:.0f} secondes (estimation)")

    # Créer l'animation 3D directement
    mp4_path = create_animation(
        frames_dir=str(frames_dir),
        output_file=str(out_dir / 'antenna_precise_3d.mp4'),
        framerate=10,
        mode='3d',
        sim=sim,
        field='E',  # Champ électrique pour l'antenne
        nsteps=nsteps,
        frame_interval=frame_interval,
        sx=4, sy=4, sz=4,  # Sous-échantillonnage pour performance
        source_type='antenna',
        omega=omega,
        feed_pos=(nx//2, ny//2, nz//2),
        amplitude=0.01,  # Amplitude très réduite pour éviter l'overflow
        envelope_type='gaussian',
        t0=t_total / 2,  # Milieu de la simulation
        width=t_total / 20  # Largeur plus étroite pour éviter l'accumulation
    )

    if mp4_path:
        print(f"Animation 3D créée: {mp4_path}")
    else:
        print("Échec de création de l'animation.")

if __name__ == '__main__':
    main()
