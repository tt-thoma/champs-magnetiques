"""
Demonstration de la difference entre source continue et impulsion.

Compare les deux types de sources pour montrer l'amelioration.
"""
import numpy as np
import matplotlib.pyplot as plt

from champs_v4.fdtd_yee_3d import Yee3D

from . import results_dir


def simulate_with_source(source_type='pulse', nsteps=300):
    """Simule avec une source continue ou impulsionnelle."""

    # Configuration simple
    nx, ny, nz = 150, 100, 1
    dx = 1.0e-3
    c0 = 3e8
    dt = 0.45 * dx / (c0 * np.sqrt(2))

    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=10)

    # Interface dielectrique
    sim.epsilon_r[nx//2:, :, :] = 2.25

    # Source
    source_x = 25
    source_y = ny // 2
    freq = 8e9
    omega = 2 * np.pi * freq

    # Simulation
    snapshots = []
    snapshot_times = [50, 100, 150, 200, 250]

    if source_type == 'pulse':
        # Impulsion gaussienne (NOUVEAU)
        t0 = 40 * dt
        spread = 15 * dt

        for n in range(nsteps):
            t = n * dt
            pulse = np.exp(-((t - t0) / spread)**2) * np.sin(omega * t)
            sim.Ez[source_x, source_y, 0] += pulse
            sim.step()

            if n in snapshot_times:
                snapshots.append(sim.Ez[:, :, 0].copy())

    else:  # 'continuous'
        # Source sinusoidale continue (ANCIEN)
        t0 = 60 * dt
        width = 30 * dt

        for n in range(nsteps):
            t = n * dt
            envelope = np.exp(-0.5 * ((t - t0) / width)**2)
            continuous = envelope * np.sin(omega * t)
            sim.Ez[source_x, source_y, 0] += continuous
            sim.step()

            if n in snapshot_times:
                snapshots.append(sim.Ez[:, :, 0].copy())

    return snapshots, snapshot_times


def main():
    print("=" * 70)
    print(" " * 15 + "COMPARAISON : Sources Continue vs Impulsion")
    print("=" * 70)
    print()
    print("Cette demo compare :")
    print("  1. SOURCE CONTINUE (ancienne methode)")
    print("     - Onde sinusoidale avec enveloppe gaussienne large")
    print("     - Plusieurs cycles d'oscillation")
    print("     - Difficile de voir la propagation")
    print()
    print("  2. SOURCE IMPULSION (nouvelle methode)")
    print("     - Impulsion gaussienne courte")
    print("     - Un seul paquet d'onde")
    print("     - Propagation claire et visible")
    print()

    # Simuler les deux types
    print("Simulation avec source CONTINUE...")
    snapshots_cont, times = simulate_with_source('continuous', nsteps=300)

    print("Simulation avec source IMPULSION...")
    snapshots_pulse, _ = simulate_with_source('pulse', nsteps=300)

    print()
    print("Generation de la comparaison visuelle...")

    # Creer figure de comparaison
    fig, axes = plt.subplots(5, 2, figsize=(14, 16))

    vmax = 0.3

    for i, (snap_cont, snap_pulse, t) in enumerate(zip(snapshots_cont, snapshots_pulse, times)):
        # Colonne 1 : Continue
        im1 = axes[i, 0].imshow(snap_cont.T, origin='lower',
                                cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                                extent=[0, 150, 0, 100], aspect='auto')
        axes[i, 0].axvline(75, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[i, 0].set_ylabel(f't = pas {t}', fontsize=10, fontweight='bold')
        if i == 0:
            axes[i, 0].set_title('SOURCE CONTINUE (ancienne)',
                                fontsize=12, fontweight='bold')
        axes[i, 0].set_xlim(0, 150)
        axes[i, 0].set_ylim(0, 100)

        # Colonne 2 : Impulsion
        im2 = axes[i, 1].imshow(snap_pulse.T, origin='lower',
                                cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                                extent=[0, 150, 0, 100], aspect='auto')
        axes[i, 1].axvline(75, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)
        if i == 0:
            axes[i, 1].set_title('SOURCE IMPULSION (nouvelle)',
                                fontsize=12, fontweight='bold')
        axes[i, 1].set_xlim(0, 150)
        axes[i, 1].set_ylim(0, 100)

    # Supprimer labels x sauf derniere ligne
    for i in range(4):
        axes[i, 0].set_xticks([])
        axes[i, 1].set_xticks([])

    axes[4, 0].set_xlabel('X (cellules)', fontsize=10)
    axes[4, 1].set_xlabel('X (cellules)', fontsize=10)

    # Ajouter colorbar
    fig.colorbar(im2, ax=axes[:, 1], label='Ez (V/m)',
                shrink=0.6, pad=0.02)

    plt.suptitle('Comparaison : Propagation avec differents types de sources',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Sauvegarder
    output_dir = results_dir / 'source_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / 'continuous_vs_pulse.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure sauvegardee : {save_path}")
    plt.close()

    # Creer aussi un graphe du signal source
    print()
    print("Generation du graphe des signaux sources...")

    dt = 0.45 * 1e-3 / (3e8 * np.sqrt(2))
    time_steps = np.arange(0, 150)
    time_ps = time_steps * dt * 1e12

    freq = 8e9
    omega = 2 * np.pi * freq

    # Signal continu
    t0_cont = 60 * dt
    width_cont = 30 * dt
    signal_cont = np.exp(-0.5 * ((time_steps * dt - t0_cont) / width_cont)**2) * \
                  np.sin(omega * time_steps * dt)

    # Signal impulsion
    t0_pulse = 40 * dt
    spread_pulse = 15 * dt
    signal_pulse = np.exp(-((time_steps * dt - t0_pulse) / spread_pulse)**2) * \
                   np.sin(omega * time_steps * dt)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Signal continu
    axes[0].plot(time_ps, signal_cont, 'b-', linewidth=2)
    axes[0].fill_between(time_ps, signal_cont, alpha=0.3)
    axes[0].axhline(0, color='k', linestyle='-', linewidth=0.5)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title('SOURCE CONTINUE - Enveloppe large, plusieurs cycles',
                     fontsize=12, fontweight='bold')
    axes[0].set_xlim(0, time_ps[-1])

    # Signal impulsion
    axes[1].plot(time_ps, signal_pulse, 'r-', linewidth=2)
    axes[1].fill_between(time_ps, signal_pulse, alpha=0.3, color='red')
    axes[1].axhline(0, color='k', linestyle='-', linewidth=0.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('Temps (ps)', fontsize=11)
    axes[1].set_ylabel('Amplitude', fontsize=11)
    axes[1].set_title('SOURCE IMPULSION - Paquet court et localise',
                     fontsize=12, fontweight='bold')
    axes[1].set_xlim(0, time_ps[-1])

    plt.tight_layout()
    save_path_signal = output_dir / 'signal_comparison.png'
    plt.savefig(save_path_signal, dpi=150, bbox_inches='tight')
    print(f"Signaux sauvegardes : {save_path_signal}")
    plt.close()

    print()
    print("=" * 70)
    print("CONCLUSION :")
    print()
    print("SOURCE CONTINUE (ancienne) :")
    print("  - Plusieurs cycles oscillants")
    print("  - Onde etendue dans l'espace")
    print("  - Difficile de distinguer reflexion/transmission")
    print("  - Peut creer interferences complexes")
    print()
    print("SOURCE IMPULSION (nouvelle) :")
    print("  + Paquet d'onde compact")
    print("  + Propagation CLAIRE et VISIBLE")
    print("  + Facile de voir reflexion et refraction")
    print("  + Meilleure pedagogie")
    print()
    print(f"Resultats dans : {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
