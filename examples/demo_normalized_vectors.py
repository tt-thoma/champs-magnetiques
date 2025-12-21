"""
Demonstration du mode de visualisation NORMALISE.

Tous les vecteurs ont la meme longueur (normalises) mais sont colores
selon leur magnitude originale. Ideal pour voir les directions du champ
sans que les zones de faible amplitude soient invisibles.
"""
import numpy as np
import matplotlib.pyplot as plt

from champs_v4.fdtd_yee_3d import Yee3D
from champs_v4.visualization.vector_field_viz import VectorFieldVisualizer

from . import results_dir


def main():
    print("=" * 70)
    print(" " * 15 + "Mode de visualisation NORMALISE")
    print("=" * 70)
    print()

    # Configuration 2D avec interface dielectrique
    nx, ny, nz = 180, 180, 1
    dx = 1.0e-3  # 1 mm

    # Calculer dt avec CFL
    c = 3e8  # vitesse de la lumiere
    dt = 0.9 * dx / (c * np.sqrt(3))

    print("Configuration :")
    print(f"  Grille : {nx}x{ny}x{nz}")
    print(f"  Resolution : dx = {dx*1e3} mm")
    print(f"  Pas de temps : dt = {dt*1e12:.3f} ps")
    print()

    # Creer simulation
    sim = Yee3D(nx, ny, nz, dx, dt)

    # Materiau : interface air/verre + obstacle conducteur
    print("Materiaux :")
    print("  - Air (gauche) : epsilon_r = 1")
    print("  - Verre (droite) : epsilon_r = 4")
    print("  - Obstacle circulaire conducteur")
    print()

    # Interface verticale a x = nx//2
    sim.epsilon_r[nx//2:, :, :] = 4.0

    # Obstacle conducteur circulaire
    cx, cy = nx // 2, ny // 2
    radius = 15
    for i in range(nx):
        for j in range(ny):
            if (i - cx)**2 + (j - cy)**2 < radius**2:
                sim.sigma[i, j, :] = 1e6  # Conducteur

    # Source : impulsion gaussienne (plus realiste)
    freq = 5e9  # 5 GHz (frequence centrale)
    wavelength = 3e8 / freq

    src_x, src_y = nx // 4, ny // 2
    print(f"Source :")
    print(f"  Position : ({src_x}, {src_y})")
    print(f"  Type : Impulsion gaussienne")
    print(f"  Frequence centrale : {freq/1e9} GHz")
    print(f"  Longueur d'onde : {wavelength*1e3:.2f} mm")
    print()

    # Parametres de l'impulsion gaussienne
    t0 = 60 * dt  # Centre de l'impulsion
    spread = 20 * dt  # Largeur de l'impulsion

    # Simulation
    n_steps = 500
    print(f"Simulation de {n_steps} pas...")

    for step in range(n_steps):
        if step % 100 == 0:
            print(f"  Pas {step}/{n_steps}")

        # Source impulsionnelle gaussienne sur Ez (mode TM)
        t = step * dt
        pulse = np.exp(-((t - t0) / spread)**2)
        sim.Ez[src_x, src_y, 0] += pulse

        # Un pas de simulation complet
        sim.step()

    print()
    print("Simulation terminee, generation des visualisations...")
    print()

    # Repertoire de sortie
    output_dir = results_dir / 'normalized_demo'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Creer visualiseur
    viz = VectorFieldVisualizer(sim, field='auto', z_index=0)

    # ========== Comparaison : Normal vs Normalise ==========
    print("1. Comparaison Quiver normal vs Vecteurs normalises...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Mode standard : grandes fleches cachent petites fleches
    viz.plot_quiver(axes[0], step=5, show_magnitude_bg=True)
    axes[0].set_title('Mode STANDARD - Fleches proportionnelles')

    # Mode normalise : toutes les fleches meme taille
    viz.plot_normalized(axes[1], step=5, arrow_scale=3.5, show_magnitude_bg=True)
    axes[1].set_title('Mode NORMALISE - Toutes fleches meme taille')

    plt.tight_layout()
    save_path = output_dir / 'comparison_standard_vs_normalized.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Sauvegarde : {save_path}")
    plt.close()

    # ========== Differentes densites ==========
    print("2. Visualisation normalisee avec differentes densites...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()

    steps = [3, 5, 7, 10]
    scales = [4.0, 3.5, 3.0, 2.5]

    for i, (step, scale) in enumerate(zip(steps, scales)):
        viz.plot_normalized(axes[i], step=step, arrow_scale=scale,
                           show_magnitude_bg=True, cmap='plasma')
        axes[i].set_title(f'Pas={step}, Echelle={scale}')

    plt.tight_layout()
    save_path = output_dir / 'normalized_different_densities.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Sauvegarde : {save_path}")
    plt.close()

    # ========== Differentes colormaps ==========
    print("3. Visualisation normalisee avec differentes colormaps...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()

    cmaps = ['jet', 'viridis', 'plasma', 'turbo']

    for i, cmap in enumerate(cmaps):
        viz.plot_normalized(axes[i], step=5, arrow_scale=3.5,
                           show_magnitude_bg=True, cmap=cmap)
        axes[i].set_title(f'Colormap: {cmap}')

    plt.tight_layout()
    save_path = output_dir / 'normalized_different_colormaps.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Sauvegarde : {save_path}")
    plt.close()

    # ========== Mode seul optimal ==========
    print("4. Mode normalise optimal (grande taille)...")

    fig, ax = plt.subplots(figsize=(12, 10))
    viz.plot_normalized(ax, step=4, arrow_scale=3.5,
                       show_magnitude_bg=True, cmap='turbo')

    save_path = output_dir / 'normalized_optimal.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"   Sauvegarde : {save_path}")
    plt.close()

    print()
    print("=" * 70)
    print("AVANTAGES du mode NORMALISE :")
    print("  - Toutes les directions visibles, meme en zones faibles")
    print("  - Pas de masquage par les vecteurs de forte amplitude")
    print("  - Couleur indique la magnitude originale")
    print("  - Ideal pour analyser la topologie du champ")
    print()
    print(f"Fichiers sauvegardes dans : {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
