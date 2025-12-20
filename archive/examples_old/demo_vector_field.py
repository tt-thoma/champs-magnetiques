"""
Exemple de visualisations vectorielles des champs EM.
Compare les 3 modes : streamlines, quiver, et hybride.
"""
import numpy as np
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from champs_v4.fdtd_yee_3d import Yee3D
from champs_v4.visualization.vector_field_viz import VectorFieldVisualizer, compare_visualizations


def main():
    print("=" * 70)
    print(" Demonstration des visualisations vectorielles ".center(70))
    print("=" * 70)
    
    # Grille 2D avec interface dielectrique
    nx, ny, nz = 160, 160, 1
    dx = 1e-3  # 1 mm
    c0 = 3e8
    dt = 0.45 * dx / (c0 * np.sqrt(2))
    
    print(f"\nConfiguration :")
    print(f"  Grille : {nx}x{ny}x{nz}")
    print(f"  Resolution : dx = {dx*1e3:.1f} mm")
    
    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=10)
    
    # Creer une interface dielectrique interessante
    epsilon_r = np.ones((nx, ny, nz))
    sigma = np.zeros((nx, ny, nz))
    
    # Interface verticale air/verre
    epsilon_r[nx//2:, :, :] = 4.0
    
    # Ajouter un obstacle circulaire conducteur
    cx, cy = nx//2, ny//2
    radius = 15
    y_grid, x_grid = np.ogrid[:nx, :ny]
    circle_mask = (x_grid - cx)**2 + (y_grid - cy)**2 <= radius**2
    sigma[circle_mask, 0] = 1e6  # Conducteur
    
    sim.set_materials(epsilon_r, sigma)
    
    print(f"\nMmateriaux :")
    print(f"  - Air (gauche) : epsilon_r = 1")
    print(f"  - Verre (droite) : epsilon_r = 4")
    print(f"  - Obstacle circulaire conducteur au centre")
    
    # Source ponctuelle
    source_pos = (30, ny//2, 0)
    freq = 8e9  # 8 GHz
    omega = 2 * np.pi * freq
    
    print(f"\nSource :")
    print(f"  Position : {source_pos[:2]}")
    print(f"  Frequence : {freq/1e9:.1f} GHz")
    
    # Simulation jusqu'a regime etabli
    nsteps = 400
    t0 = 80 * dt
    width = 30 * dt
    
    print(f"\nSimulation de {nsteps} pas...")
    
    for n in range(nsteps):
        t = n * dt
        envelope = np.exp(-0.5 * ((t - t0) / width) ** 2)
        source_value = 0.3 * envelope * np.sin(omega * t)
        
        sim.Ez[source_pos] += source_value
        sim.step()
        
        if n % 100 == 0:
            print(f"  Pas {n}/{nsteps}")
    
    print("\nSimulation terminee, generation des visualisations...")
    
    # Creer dossier de sortie
    out_dir = parent_dir / 'champs_v4' / 'results' / 'vector_field_demo'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Comparaison des 3 modes
    print("\n1. Comparaison des 3 modes de visualisation...")
    compare_visualizations(sim, field='auto', z_index=0,
                          save_path=out_dir / 'comparison_3modes.png')
    print(f"   Sauvegarde : {out_dir / 'comparison_3modes.png'}")
    
    # 2. Visualisations individuelles detaillees
    print("\n2. Mode Streamlines (lignes de champ)...")
    viz = VectorFieldVisualizer(sim, field='auto', z_index=0)
    
    # Mode Streamlines
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 9))
    viz.plot_streamlines(ax, density=2.0, color_by_magnitude=True, downsample=1)
    
    # Marquer l'interface et l'obstacle
    ax.axvline(x=nx//2, color='yellow', linestyle='--', linewidth=2, alpha=0.5, label='Interface')
    circle = plt.Circle((cy, cx), radius, fill=False, edgecolor='red', linewidth=2, label='Obstacle')
    ax.add_patch(circle)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'streamlines_detailed.png', dpi=150)
    plt.close()
    print(f"   Sauvegarde : {out_dir / 'streamlines_detailed.png'}")
    
    # Mode Quiver
    print("\n3. Mode Quiver (fleches vectorielles)...")
    fig, ax = plt.subplots(figsize=(10, 9))
    viz.plot_quiver(ax, step=4, scale=30, show_magnitude_bg=True)
    
    ax.axvline(x=nx//2, color='cyan', linestyle='--', linewidth=2, alpha=0.5)
    circle = plt.Circle((cy, cx), radius, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(circle)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'quiver_detailed.png', dpi=150)
    plt.close()
    print(f"   Sauvegarde : {out_dir / 'quiver_detailed.png'}")
    
    # Mode Hybride
    print("\n4. Mode Hybride (magnitude + streamlines + vecteurs)...")
    fig, ax = plt.subplots(figsize=(10, 9))
    viz.plot_hybrid(ax, streamline_density=1.5, quiver_step=8)
    
    ax.axvline(x=nx//2, color='lime', linestyle='--', linewidth=2, alpha=0.5)
    circle = plt.Circle((cy, cx), radius, fill=False, edgecolor='yellow', linewidth=2)
    ax.add_patch(circle)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'hybrid_detailed.png', dpi=150)
    plt.close()
    print(f"   Sauvegarde : {out_dir / 'hybrid_detailed.png'}")
    
    # 5. Champ complementaire (force l'autre champ)
    print("\n5. Champ complementaire (mode force)...")
    # Forcer E si on a visualise H, ou H si on a visualise E
    other_field = 'E' if viz.field == 'H' else 'H'
    viz_other = VectorFieldVisualizer(sim, field=other_field, z_index=0)
    fig, ax = plt.subplots(figsize=(10, 9))
    viz_other.plot_streamlines(ax, density=1.8, color_by_magnitude=True)
    
    ax.axvline(x=nx//2, color='yellow', linestyle='--', linewidth=2, alpha=0.5)
    circle = plt.Circle((cy, cx), radius, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(circle)
    
    plt.tight_layout()
    plt.savefig(out_dir / f'{other_field}_field_complementary.png', dpi=150)
    plt.close()
    print(f"   Sauvegarde : {out_dir / f'{other_field}_field_complementary.png'}")
    
    print("\n" + "=" * 70)
    print(" TERMINEE ".center(70))
    print("=" * 70)
    print(f"\nToutes les visualisations sont dans : {out_dir}")
    print("\nFichiers crees :")
    print("  1. comparison_3modes.png     - Comparaison cote a cote")
    print("  2. streamlines_detailed.png  - Lignes de champ (auto-detecte)")
    print("  3. quiver_detailed.png       - Vecteurs (auto-detecte)")
    print("  4. hybrid_detailed.png       - Vue hybride (auto-detecte)")
    print("  5. X_field_complementary.png - Champ complementaire")
    
    print("\nCaracteristiques des modes :")
    print("  STREAMLINES : Montre circulation et flux, ideal pour topologie")
    print("  QUIVER      : Montre direction et intensite locale, quantitatif")
    print("  HYBRID      : Combine les avantages, meilleure vue d'ensemble")


if __name__ == '__main__':
    main()
