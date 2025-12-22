"""
Demonstration SIMPLE de propagation d'onde electromagnetique.

Simulation claire d'une onde se propageant dans un milieu,
rencontrant une interface dielectrique (air/verre).
"""
import numpy as np
import matplotlib.pyplot as plt

from champs_v4.fdtd_yee_3d import Yee3D
from champs_v4.visualization.vector_field_viz import VectorFieldVisualizer

from . import results_dir


def main():
    print("=" * 70)
    print(" " * 15 + "PROPAGATION D'ONDE ELECTROMAGNETIQUE")
    print("=" * 70)
    print()
    print("Cette simulation montre :")
    print("  - Une IMPULSION electromagnetique (pas une onde continue)")
    print("  - Se propageant de gauche a droite")
    print("  - Rencontrant une interface AIR/VERRE")
    print("  - REFLEXION partielle + TRANSMISSION")
    print()

    # Configuration simple
    nx, ny, nz = 200, 120, 1
    dx = 1.0e-3  # 1 mm

    c = 3e8
    dt = 0.9 * dx / (c * np.sqrt(3))

    print("Configuration :")
    print(f"  Grille : {nx}x{ny}x{nz} cellules")
    print(f"  Taille cellule : {dx*1e3} mm")
    print(f"  Vitesse lumiere : c = {c/1e8:.0f}e8 m/s")
    print()

    # Creer simulation
    sim = Yee3D(nx, ny, nz, dx, dt)

    # Materiau : interface verticale simple
    interface_x = nx // 2
    epsilon_verre = 2.25  # indice n=1.5

    sim.epsilon_r[interface_x:, :, :] = epsilon_verre

    print("Materiaux :")
    print(f"  Gauche (x < {interface_x}) : AIR (epsilon_r = 1.0, n = 1.0)")
    print(f"  Droite (x > {interface_x}) : VERRE (epsilon_r = {epsilon_verre}, n = {np.sqrt(epsilon_verre):.2f})")
    print()

    # Source : impulsion gaussienne
    freq = 3e9  # 3 GHz
    wavelength = c / freq

    src_x = 30  # Loin de l'interface
    src_y = ny // 2

    print(f"Source :")
    print(f"  Type : Impulsion gaussienne (1 seule impulsion)")
    print(f"  Position : x={src_x}, y={src_y}")
    print(f"  Frequence centrale : {freq/1e9} GHz")
    print(f"  Longueur d'onde (air) : {wavelength*1e3:.1f} mm")
    print(f"  Longueur d'onde (verre) : {wavelength*1e3/np.sqrt(epsilon_verre):.1f} mm")
    print()

    # Parametres impulsion
    t0 = 40 * dt
    spread = 12 * dt

    # Simulation
    n_steps = 600
    print(f"Simulation en cours : {n_steps} pas de temps...")
    print()

    for step in range(n_steps):
        if step % 150 == 0:
            print(f"  Pas {step}/{n_steps} (t = {step*dt*1e12:.1f} ps)")

        # Source impulsionnelle
        t = step * dt
        pulse = np.exp(-((t - t0) / spread)**2) * np.sin(2 * np.pi * freq * t)
        sim.Ez[src_x, src_y, 0] += pulse

        sim.step()

    print()
    print("Simulation terminee !")
    print()

    # Visualisation
    output_dir = results_dir / 'demo_simple'
    output_dir.mkdir(parents=True, exist_ok=True)

    viz = VectorFieldVisualizer(sim, field='auto', z_index=0)

    # Vue 1 : Vecteurs normalises (direction claire)
    print("Visualisation 1 : Vecteurs normalises...")
    fig, ax = plt.subplots(figsize=(14, 7))
    viz.plot_normalized(ax, step=6, arrow_scale=4.0, show_magnitude_bg=True, cmap='turbo')

    # Ajouter ligne interface
    ax.axvline(interface_x, color='white', linestyle='--', linewidth=2, alpha=0.8, label='Interface air/verre')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.8)
    ax.set_title('VECTEURS NORMALISES - Direction du champ H (mode TM)', fontsize=14, fontweight='bold')

    save_path = output_dir / 'propagation_vecteurs_normalises.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Sauvegarde : {save_path}")
    plt.close()

    # Vue 2 : Streamlines (flux du champ)
    print("Visualisation 2 : Lignes de champ...")
    fig, ax = plt.subplots(figsize=(14, 7))
    viz.plot_streamlines(ax, density=1.0, color_by_magnitude=True)

    ax.axvline(interface_x, color='cyan', linestyle='--', linewidth=2, alpha=0.8, label='Interface air/verre')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.8)
    ax.set_title('LIGNES DE CHAMP - Flux du champ magnetique H', fontsize=14, fontweight='bold')

    save_path = output_dir / 'propagation_streamlines.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Sauvegarde : {save_path}")
    plt.close()

    # Vue 3 : Comparaison cote a cote
    print("Visualisation 3 : Comparaison...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    viz.plot_normalized(axes[0], step=6, arrow_scale=4.0, show_magnitude_bg=True, cmap='turbo')
    axes[0].axvline(interface_x, color='white', linestyle='--', linewidth=2, alpha=0.8)
    axes[0].set_title('Vecteurs normalises', fontsize=12, fontweight='bold')

    viz.plot_quiver(axes[1], step=8, show_magnitude_bg=True)
    axes[1].axvline(interface_x, color='white', linestyle='--', linewidth=2, alpha=0.8)
    axes[1].set_title('Vecteurs proportionnels', fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = output_dir / 'propagation_comparaison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Sauvegarde : {save_path}")
    plt.close()

    print()
    print("=" * 70)
    print("INTERPRETATION des resultats :")
    print()
    print("  1. ONDE INCIDENTE (gauche) :")
    print("     - Vecteurs pointent dans diverses directions")
    print("     - Representation du champ magnetique H tournant")
    print()
    print("  2. INTERFACE (ligne verticale) :")
    print("     - Changement d'indice : n=1.0 -> n=1.5")
    print("     - Onde ralentit dans le verre")
    print("     - Longueur d'onde reduite")
    print()
    print("  3. REFLEXION + TRANSMISSION :")
    print("     - Partie de l'onde reflechie (retour vers gauche)")
    print("     - Partie transmise (continue vers droite)")
    print("     - Conservation de l'energie")
    print()
    print("  MODE TM (Ez perpendiculaire) :")
    print("     - Champ electrique E pointe hors du plan (Ez)")
    print("     - Champ magnetique H tourne dans le plan (Hx, Hy)")
    print("     - C'est H qu'on visualise ici")
    print()
    print(f"Fichiers dans : {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
