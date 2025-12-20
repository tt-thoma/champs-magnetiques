"""
Test de l'auto-detection des champs vectoriels.
Compare E et H pour verifier que les vecteurs sont visibles.
"""
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from champs_v4.fdtd_yee_3d import Yee3D
from champs_v4.visualization.vector_field_viz import VectorFieldVisualizer, _detect_dominant_mode


def main():
    print("=" * 70)
    print(" Test Auto-Detection des Champs Vectoriels ".center(70))
    print("=" * 70)
    
    # Simulation 2D simple
    nx, ny, nz = 120, 120, 1
    dx = 1e-3
    c0 = 3e8
    dt = 0.45 * dx / (c0 * np.sqrt(2))
    
    print(f"\nGrille : {nx}x{ny}x{nz}")
    
    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=10)
    
    # Interface dielectrique
    epsilon_r = np.ones((nx, ny, nz))
    epsilon_r[nx//2:, :, :] = 4.0
    sigma = np.zeros((nx, ny, nz))
    sim.set_materials(epsilon_r, sigma)
    
    # Source ponctuelle
    source_pos = (25, ny//2, 0)
    freq = 10e9
    omega = 2 * np.pi * freq
    
    print("Source ponctuelle, frequence 10 GHz")
    print("Simulation de 300 pas...")
    
    # Simulation courte
    t0 = 60 * dt
    width = 25 * dt
    
    for n in range(300):
        t = n * dt
        envelope = np.exp(-0.5 * ((t - t0) / width) ** 2)
        source_value = 0.4 * envelope * np.sin(omega * t)
        
        sim.Ez[source_pos] += source_value
        sim.step()
    
    print("Simulation terminee\\n")
    
    # Detection du mode
    print("=" * 70)
    print(" Analyse des composantes du champ ".center(70))
    print("=" * 70)
    
    mode = _detect_dominant_mode(sim, 0)
    
    from champs_v4.visualization.vector_field_viz import _centered_E_plane, _centered_H_plane
    
    Ez_mag = np.abs(sim.Ez[:, :, 0]).max()
    Hz_mag = np.abs(sim.Hz[:, :, 0]).max()
    
    Ex_c, Ey_c = _centered_E_plane(sim, 0)
    Exy_mag = np.sqrt(Ex_c**2 + Ey_c**2).max()
    
    Hx_c, Hy_c = _centered_H_plane(sim, 0)
    Hxy_mag = np.sqrt(Hx_c**2 + Hy_c**2).max()
    
    print(f"\\nMagnitudes des composantes :")
    print(f"  Ez (perpendiculaire) : {Ez_mag:.3e}")
    print(f"  E_xy (dans le plan)  : {Exy_mag:.3e}")
    print(f"  Hz (perpendiculaire) : {Hz_mag:.3e}")
    print(f"  H_xy (dans le plan)  : {Hxy_mag:.3e}")
    print(f"\\nMode detecte : {mode}")
    
    if mode == 'TM':
        print("  -> Ez domine : VISUALISER H dans le plan (Hx, Hy)")
    elif mode == 'TE':
        print("  -> Hz domine : VISUALISER E dans le plan (Ex, Ey)")
    else:
        print("  -> Mode mixte")
    
    # Test des visualisations
    print("\\n" + "=" * 70)
    print(" Test des visualisations ".center(70))
    print("=" * 70)
    
    out_dir = parent_dir / 'champs_v4' / 'results' / 'test_autodetect'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Mode AUTO (recommande)
    print("\\n1. Mode AUTO (recommande) :")
    viz_auto = VectorFieldVisualizer(sim, field='auto', z_index=0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    viz_auto.plot_hybrid(ax, streamline_density=1.5, quiver_step=8)
    ax.axvline(x=nx//2, color='lime', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_title(f'Mode AUTO : Visualisation de {viz_auto.field}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'auto_detection.png', dpi=120)
    plt.close()
    print(f"   Champ affiche : {viz_auto.field}")
    print(f"   Sauvegarde : auto_detection.png")
    
    # 2. Forcer E (mauvais choix si Ez domine)
    print("\\n2. Mode force E (peut etre faible) :")
    viz_E = VectorFieldVisualizer(sim, field='E', z_index=0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    viz_E.plot_hybrid(ax, streamline_density=1.5, quiver_step=8)
    ax.axvline(x=nx//2, color='lime', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_title('Mode force E : E dans le plan (peut etre faible)', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / 'forced_E.png', dpi=120)
    plt.close()
    print(f"   Sauvegarde : forced_E.png")
    
    # 3. Forcer H (bon choix si Ez domine)
    print("\\n3. Mode force H (bon choix si mode TM) :")
    viz_H = VectorFieldVisualizer(sim, field='H', z_index=0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    viz_H.plot_hybrid(ax, streamline_density=1.5, quiver_step=8)
    ax.axvline(x=nx//2, color='lime', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_title('Mode force H : H dans le plan', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / 'forced_H.png', dpi=120)
    plt.close()
    print(f"   Sauvegarde : forced_H.png")
    
    # Comparaison cote a cote
    print("\\n4. Comparaison E vs H cote a cote :")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    viz_E_comp = VectorFieldVisualizer(sim, field='E', z_index=0)
    viz_E_comp.plot_hybrid(axes[0], streamline_density=1.2, quiver_step=8)
    axes[0].axvline(x=nx//2, color='yellow', linestyle='--', linewidth=2, alpha=0.5)
    axes[0].set_title('Champ E dans le plan (Ex, Ey)', fontsize=12, fontweight='bold')
    
    viz_H_comp = VectorFieldVisualizer(sim, field='H', z_index=0)
    viz_H_comp.plot_hybrid(axes[1], streamline_density=1.2, quiver_step=8)
    axes[1].axvline(x=nx//2, color='yellow', linestyle='--', linewidth=2, alpha=0.5)
    axes[1].set_title('Champ H dans le plan (Hx, Hy)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'comparison_E_vs_H.png', dpi=120)
    plt.close()
    print(f"   Sauvegarde : comparison_E_vs_H.png")
    
    print("\\n" + "=" * 70)
    print(" RESULTATS ".center(70))
    print("=" * 70)
    print(f"\\nFichiers sauvegardes dans : {out_dir}")
    print("\\nCOMPAREZ les images :")
    print("  - auto_detection.png    : Mode AUTO (recommande)")
    print("  - forced_E.png          : E force (peut etre faible)")
    print("  - forced_H.png          : H force")
    print("  - comparison_E_vs_H.png : E et H cote a cote")
    
    if mode == 'TM':
        print("\\nCONCLUSION : Mode TM (Ez domine)")
        print("  -> Les vecteurs H sont bien visibles")
        print("  -> Les vecteurs E dans le plan sont faibles")
        print("  => Utilisez field='auto' ou field='H'")
    else:
        print("\\nCONCLUSION : Autre mode")
        print("  => Comparez les images pour choisir le meilleur champ")


if __name__ == '__main__':
    main()
