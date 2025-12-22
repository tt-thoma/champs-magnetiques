#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Démonstration de l'analyse de Fourier avec les sondes.

Ce script montre comment utiliser les sondes pour analyser le contenu
fréquentiel des champs électromagnétiques enregistrés.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from champs_v4.fdtd_yee_3d import Yee3D


def main():
    """Simulation avec analyse fréquentielle des sondes."""
    
    print("="*70)
    print(" ANALYSE DE FOURIER DES SONDES ÉLECTROMAGNÉTIQUES")
    print("="*70)
    
    # === Configuration ===
    nx, ny, nz = 150, 100, 1
    dx = 1e-3  # 1 mm
    c = 3e8
    dt = dx / (c * np.sqrt(2)) * 0.5
    
    print(f"\nGrille: {nx} x {ny} x {nz}")
    print(f"dx = {dx*1e3:.1f} mm, dt = {dt*1e12:.2f} ps")
    
    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=10)
    
    # === Sondes ===
    print("\n=== Placement des sondes ===")
    
    source_x = 30
    
    probes_config = [
        (source_x + 5, ny//2, 0, "near_source"),
        (source_x + 30, ny//2, 0, "probe_30mm"),
        (source_x + 60, ny//2, 0, "probe_60mm"),
        (nx - 25, ny//2, 0, "before_pml"),
    ]
    
    for x, y, z, name in probes_config:
        sim.add_probe((x, y, z), name, record_E=True, record_H=True, record_energy=True)
        print(f"  ✓ {name:15s} à x={x}")
    
    # === Source multi-fréquence ===
    print("\n=== Source composite ===")
    
    # Mélange de 3 fréquences
    freqs = [3e9, 5e9, 8e9]  # 3, 5, 8 GHz
    amplitudes = [1.0, 0.7, 0.4]
    
    print("Fréquences injectées:")
    for f, a in zip(freqs, amplitudes):
        wavelength = c / f
        print(f"  - {f*1e-9:.1f} GHz (λ={wavelength*1e3:.1f} mm), amplitude={a:.1f}")
    
    omegas = [2 * np.pi * f for f in freqs]
    
    # === Simulation ===
    nsteps = 1200
    print(f"\nSimulation: {nsteps} pas")
    
    # Enveloppe gaussienne pour éviter les transitoires
    t0 = 100 * dt
    sigma_t = 50 * dt
    
    for step in range(nsteps):
        t = step * dt
        
        # Signal composite avec enveloppe
        envelope = np.exp(-0.5 * np.square((t - t0) / sigma_t))
        signal = sum(a * np.sin(omega * t) for a, omega in zip(amplitudes, omegas))
        source_value = envelope * signal * 0.05  # Amplitude modérée
        
        # Injection
        sim.Ez[source_x, ny//2, 0] = source_value
        sim.step()
        
        if step % 200 == 0:
            print(f"  Pas {step}/{nsteps}")
    
    print("✓ Simulation terminée")
    
    # === Analyse de Fourier ===
    print("\n" + "="*70)
    print(" ANALYSE FRÉQUENTIELLE")
    print("="*70)
    
    probe_data = sim.get_probe_data()
    
    # Analyser chaque sonde
    fft_results = {}
    for name in ['near_source', 'probe_30mm', 'probe_60mm', 'before_pml']:
        if name in probe_data:
            print(f"\n{name}:")
            
            # Récupérer la sonde
            probe = sim.probe_manager.get_probe(name)
            
            # Calculer FFT
            fft_data = probe.compute_fft('Ez', window='hann')
            fft_results[name] = fft_data
            
            # Trouver les pics
            dominant_freqs = probe.get_dominant_frequencies('Ez', n_peaks=5)
            
            print("  Fréquences dominantes détectées:")
            for i, (freq, amp) in enumerate(dominant_freqs[:3], 1):
                print(f"    {i}. {freq*1e-9:.2f} GHz - Amplitude: {amp:.2e}")
    
    # === Visualisation ===
    print("\n=== Génération des graphiques ===")
    
    # Graphique 1: Signaux temporels
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    for idx, name in enumerate(['near_source', 'probe_30mm', 'probe_60mm', 'before_pml']):
        if name in probe_data:
            data = probe_data[name]
            t_ns = data['time'] * 1e9
            Ez = data['Ez']
            
            axes[idx].plot(t_ns, Ez, 'b-', linewidth=0.6)
            axes[idx].set_ylabel('Ez (V/m)', fontsize=10)
            axes[idx].set_title(f'Signal temporel - {name}', fontsize=11, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            
            if idx == 3:
                axes[idx].set_xlabel('Temps (ns)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('fft_probe_signals.png', dpi=150, bbox_inches='tight')
    print("✓ fft_probe_signals.png")
    plt.close()
    
    # Graphique 2: Spectres de Fourier
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, name in enumerate(['near_source', 'probe_30mm', 'probe_60mm', 'before_pml']):
        if name in fft_results:
            fft_data = fft_results[name]
            freqs_GHz = fft_data['frequencies'] * 1e-9
            amplitude = fft_data['amplitude']
            
            # Limiter à 0-15 GHz pour la visibilité
            mask = freqs_GHz <= 15
            
            axes[idx].plot(freqs_GHz[mask], amplitude[mask], 'b-', linewidth=1)
            axes[idx].set_xlabel('Fréquence (GHz)', fontsize=10)
            axes[idx].set_ylabel('Amplitude', fontsize=10)
            axes[idx].set_title(f'Spectre FFT - {name}', fontsize=11, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            
            # Marquer les fréquences injectées
            for f_inj in freqs:
                axes[idx].axvline(f_inj*1e-9, color='r', linestyle='--', 
                                alpha=0.5, linewidth=1)
            
            # Ajouter les pics détectés
            probe = sim.probe_manager.get_probe(name)
            dominant = probe.get_dominant_frequencies('Ez', n_peaks=3)
            for f, amp in dominant:
                if f*1e-9 <= 15:
                    axes[idx].plot(f*1e-9, amp, 'ro', markersize=8)
    
    plt.tight_layout()
    plt.savefig('fft_probe_spectra.png', dpi=150, bbox_inches='tight')
    print("✓ fft_probe_spectra.png")
    plt.close()
    
    # Graphique 3: Comparaison des spectres
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['blue', 'green', 'orange', 'red']
    for name, color in zip(['near_source', 'probe_30mm', 'probe_60mm', 'before_pml'], colors):
        if name in fft_results:
            fft_data = fft_results[name]
            freqs_GHz = fft_data['frequencies'] * 1e-9
            amplitude = fft_data['amplitude']
            
            mask = freqs_GHz <= 15
            ax.plot(freqs_GHz[mask], amplitude[mask], color=color, 
                   linewidth=1.5, label=name, alpha=0.7)
    
    # Marquer les fréquences sources
    for f_inj, amp_inj in zip(freqs, amplitudes):
        ax.axvline(f_inj*1e-9, color='red', linestyle='--', 
                  alpha=0.3, linewidth=2)
        ax.text(f_inj*1e-9, ax.get_ylim()[1]*0.9, 
               f'{f_inj*1e-9:.1f} GHz', 
               rotation=90, va='top', ha='right', fontsize=9, color='red')
    
    ax.set_xlabel('Fréquence (GHz)', fontsize=11)
    ax.set_ylabel('Amplitude normalisée', fontsize=11)
    ax.set_title('Comparaison des spectres - Propagation multi-fréquence', 
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 15)
    
    plt.tight_layout()
    plt.savefig('fft_probe_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ fft_probe_comparison.png")
    plt.close()
    
    # Graphique 4: Densité spectrale de puissance
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for name, color in zip(['near_source', 'probe_30mm', 'probe_60mm', 'before_pml'], colors):
        if name in fft_results:
            fft_data = fft_results[name]
            freqs_GHz = fft_data['frequencies'] * 1e-9
            power = fft_data['power_spectrum']
            
            mask = freqs_GHz <= 15
            ax.semilogy(freqs_GHz[mask], power[mask], color=color, 
                       linewidth=1.5, label=name, alpha=0.7)
    
    ax.set_xlabel('Fréquence (GHz)', fontsize=11)
    ax.set_ylabel('Densité spectrale de puissance', fontsize=11)
    ax.set_title('Analyse spectrale de puissance', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0, 15)
    
    plt.tight_layout()
    plt.savefig('fft_probe_power.png', dpi=150, bbox_inches='tight')
    print("✓ fft_probe_power.png")
    
    print("\n" + "="*70)
    print(" RÉSUMÉ DE L'ANALYSE")
    print("="*70)
    
    print("\nFréquences injectées vs détectées:")
    print(f"{'Source':<15s} {'Injecté (GHz)':<20s} {'Détecté (GHz)':<20s}")
    print("-" * 55)
    
    # Comparer avec la première sonde
    probe = sim.probe_manager.get_probe('near_source')
    if probe:
        dominant = probe.get_dominant_frequencies('Ez', n_peaks=5)
        for i, f_inj in enumerate(freqs):
            detected = [f for f, _ in dominant if abs(f - f_inj) / f_inj < 0.05]
            if detected:
                print(f"Pic {i+1:<10d} {f_inj*1e-9:<20.2f} {detected[0]*1e-9:<20.2f} ✓")
            else:
                print(f"Pic {i+1:<10d} {f_inj*1e-9:<20.2f} {'Non détecté':<20s} ⚠")
    
    print("\n✓ Analyse fréquentielle complète!")


if __name__ == '__main__':
    main()
