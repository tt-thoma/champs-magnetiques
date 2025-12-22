#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation des sondes de champs électromagnétiques.

Ce script démontre comment placer des sondes dans l'espace pour
enregistrer l'évolution temporelle des champs et de l'énergie.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from champs_v4.fdtd_yee_3d import Yee3D


def main():
    """Simulation avec sondes pour observer la propagation d'une onde."""
    
    # === Configuration de la grille ===
    nx, ny, nz = 200, 100, 1  # Grille 2D (slice z mince)
    dx = 1e-3  # 1 mm
    c = 3e8
    dt = dx / (c * np.sqrt(2)) * 0.9  # Condition CFL
    
    print("=== Simulation FDTD avec sondes ===")
    print(f"Grille: {nx} x {ny} x {nz}")
    print(f"dx = {dx*1e3:.2f} mm, dt = {dt*1e12:.2f} ps")
    
    # === Création du simulateur ===
    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=10)
    
    # === Placement des sondes ===
    print("\n=== Placement des sondes ===")
    
    # Sonde au centre (source)
    sim.add_probe((nx//2, ny//2, 0), name='source', record_energy=True)
    print("✓ Sonde 'source' au centre")
    
    # Ligne de sondes horizontale pour observer la propagation
    probes_line = sim.add_line_of_probes(
        start=(20, ny//2, 0),
        end=(nx-20, ny//2, 0),
        num_probes=5,
        name_prefix='line',
        record_E=True,
        record_H=True,
        record_energy=True
    )
    print(f"✓ {len(probes_line)} sondes le long de l'axe x")
    
    # Sondes aux quatre coins (pour observer les réflexions)
    corners = [
        ((30, 30, 0), 'corner_SW'),
        ((nx-30, 30, 0), 'corner_SE'),
        ((30, ny-30, 0), 'corner_NW'),
        ((nx-30, ny-30, 0), 'corner_NE'),
    ]
    for pos, name in corners:
        sim.add_probe(pos, name=name, record_E=True, record_H=False, record_energy=True)
    print(f"✓ 4 sondes aux coins")
    
    print(f"\nTotal: {len(sim.list_probes())} sondes actives")
    
    # === Paramètres de la source ===
    source_pos = (nx//2, ny//2, 0)
    freq = 10e9  # 10 GHz
    omega = 2 * np.pi * freq
    wavelength = c / freq
    print(f"\nSource: f = {freq*1e-9:.1f} GHz, λ = {wavelength*1e3:.2f} mm")
    
    # === Simulation ===
    nsteps = 800
    print(f"\n=== Lancement de la simulation ({nsteps} pas) ===")
    
    for step in range(nsteps):
        # Source: impulsion gaussienne modulée
        t = step * dt
        pulse = np.exp(-((t - 3e-10)**2) / (1e-10)**2)
        modulation = np.sin(omega * t)
        sim.Ez[source_pos[0], source_pos[1], source_pos[2]] = pulse * modulation
        
        # Pas de simulation (les sondes sont échantillonnées automatiquement)
        sim.step()
        
        if step % 100 == 0:
            print(f"  Pas {step}/{nsteps} (t = {t*1e9:.2f} ns)")
    
    print("\n✓ Simulation terminée")
    
    # === Récupération et visualisation des données ===
    print("\n=== Traitement des données ===")
    
    # Données de toutes les sondes
    all_data = sim.get_probe_data()
    
    # === Graphique 1: Évolution temporelle au centre ===
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    source_data = all_data['source']
    t_ns = source_data['time'] * 1e9  # Convertir en ns
    
    # Champ Ez
    axes[0].plot(t_ns, source_data['Ez'], 'b-', linewidth=1)
    axes[0].set_ylabel('Ez (V/m)', fontsize=11)
    axes[0].set_title('Sonde "source" au centre de la grille', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Champ H (magnitude)
    axes[1].plot(t_ns, source_data['H_magnitude'], 'r-', linewidth=1)
    axes[1].set_ylabel('|H| (A/m)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Énergie totale
    axes[2].plot(t_ns, source_data['U_total'], 'g-', linewidth=1, label='Totale')
    axes[2].plot(t_ns, source_data['U_electric'], 'b--', linewidth=0.8, alpha=0.7, label='Électrique')
    axes[2].plot(t_ns, source_data['U_magnetic'], 'r--', linewidth=0.8, alpha=0.7, label='Magnétique')
    axes[2].set_xlabel('Temps (ns)', fontsize=11)
    axes[2].set_ylabel('Densité d\'énergie (J/m³)', fontsize=11)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('probe_example_source.png', dpi=150, bbox_inches='tight')
    print("✓ Sauvegardé: probe_example_source.png")
    
    # === Graphique 2: Comparaison des sondes en ligne ===
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    for i, color in enumerate(colors):
        probe_name = f'line_{i}'
        if probe_name in all_data:
            data = all_data[probe_name]
            t_ns = data['time'] * 1e9
            axes[0].plot(t_ns, data['Ez'], color=color, linewidth=1, label=probe_name)
            axes[1].plot(t_ns, data['U_total'], color=color, linewidth=1, label=probe_name)
    
    axes[0].set_ylabel('Ez (V/m)', fontsize=11)
    axes[0].set_title('Champs Ez le long de la ligne de sondes', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', ncol=3)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Temps (ns)', fontsize=11)
    axes[1].set_ylabel('Densité d\'énergie (J/m³)', fontsize=11)
    axes[1].set_title('Énergie le long de la ligne de sondes', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', ncol=3)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('probe_example_line.png', dpi=150, bbox_inches='tight')
    print("✓ Sauvegardé: probe_example_line.png")
    
    # === Graphique 3: Sondes aux coins ===
    fig, ax = plt.subplots(figsize=(12, 6))
    
    corner_names = ['corner_SW', 'corner_SE', 'corner_NW', 'corner_NE']
    colors_corners = ['blue', 'red', 'green', 'orange']
    
    for name, color in zip(corner_names, colors_corners):
        if name in all_data:
            data = all_data[name]
            t_ns = data['time'] * 1e9
            ax.plot(t_ns, data['Ez'], color=color, linewidth=1, label=name)
    
    ax.set_xlabel('Temps (ns)', fontsize=11)
    ax.set_ylabel('Ez (V/m)', fontsize=11)
    ax.set_title('Champs Ez aux quatre coins du domaine', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('probe_example_corners.png', dpi=150, bbox_inches='tight')
    print("✓ Sauvegardé: probe_example_corners.png")
    
    # === Sauvegarder les données ===
    from champs_v4.probes import save_probe_data
    
    probe_source = sim.probe_manager.get_probe('source')
    save_probe_data(probe_source, 'probe_source_data.npz')
    print("✓ Données sauvegardées: probe_source_data.npz")
    
    print("\n=== Terminé ===")
    print("Les graphiques ont été générés avec succès!")


if __name__ == '__main__':
    main()
