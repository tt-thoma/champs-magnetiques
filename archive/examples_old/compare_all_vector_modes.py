"""
Comparaison complete des 4 modes de visualisation vectorielle.

Ce script montre cote a cote les 4 methodes :
1. Streamlines (lignes de champ)
2. Quiver standard (fleches proportionnelles)
3. Quiver normalise (fleches uniformes)
4. Hybride (magnitude + streamlines + fleches)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ajouter le chemin parent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from champs_v4.fdtd_yee_3d import Yee3D
from champs_v4.visualization.vector_field_viz import compare_all_modes

def main():
    print("=" * 70)
    print(" " * 10 + "Comparaison des 4 modes de visualisation")
    print("=" * 70)
    print()
    
    # Configuration 2D
    nx, ny, nz = 160, 160, 1
    dx = 1.0e-3  # 1 mm
    
    # Calculer dt avec CFL
    c = 3e8
    dt = 0.9 * dx / (c * np.sqrt(3))
    
    print("Configuration :")
    print(f"  Grille : {nx}x{ny}x{nz}")
    print(f"  Resolution : dx = {dx*1e3} mm")
    print()
    
    # Creer simulation
    sim = Yee3D(nx, ny, nz, dx, dt)
    
    # Materiau : interface avec obstacle
    print("Materiaux :")
    print("  - Interface air/verre verticale")
    print("  - Obstacle conducteur circulaire")
    print()
    
    # Interface verticale
    sim.epsilon_r[nx//2:, :, :] = 4.0
    
    # Obstacle conducteur
    cx, cy = nx // 2, ny // 2
    radius = 12
    for i in range(nx):
        for j in range(ny):
            if (i - cx)**2 + (j - cy)**2 < radius**2:
                sim.sigma[i, j, :] = 1e6
    
    # Source impulsionnelle
    freq = 5e9  # 5 GHz
    src_x, src_y = nx // 4, ny // 2
    
    print(f"Source : impulsion gaussienne a ({src_x}, {src_y}), freq {freq/1e9} GHz")
    print()
    
    # Parametres impulsion
    t0 = 50 * dt
    spread = 15 * dt
    
    # Simulation
    n_steps = 400
    print(f"Simulation de {n_steps} pas...")
    
    for step in range(n_steps):
        if step % 100 == 0:
            print(f"  Pas {step}/{n_steps}")
        
        # Source impulsionnelle gaussienne
        t = step * dt
        pulse = np.exp(-((t - t0) / spread)**2)
        sim.Ez[src_x, src_y, 0] += pulse
        sim.step()
    
    print()
    print("Simulation terminee, generation de la comparaison...")
    print()
    
    # Repertoire de sortie
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'champs_v4', 'results', 'vector_comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    # Comparaison des 4 modes
    save_path = os.path.join(output_dir, 'comparison_4_modes.png')
    compare_all_modes(sim, field='auto', z_index=0, save_path=save_path)
    
    print("Comparaison des 4 modes generee :")
    print()
    print("  1. STREAMLINES : Lignes de champ continues")
    print("     - Montre la topologie globale")
    print("     - Couleur = magnitude")
    print()
    print("  2. QUIVER STANDARD : Fleches proportionnelles")
    print("     - Longueur = magnitude")
    print("     - Zones fortes dominent l'affichage")
    print()
    print("  3. QUIVER NORMALISE : Fleches uniformes")
    print("     - Toutes meme longueur")
    print("     - Couleur = magnitude originale")
    print("     - Toutes directions visibles")
    print()
    print("  4. HYBRIDE : Combinaison")
    print("     - Fond de magnitude + streamlines + fleches")
    print("     - Vue d'ensemble complete")
    print()
    print(f"Fichier sauvegarde : {save_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
