"""
Animation avec visualisation vectorielle du champ electrique.
Montre l'evolution temporelle des vecteurs du champ E.
"""
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import subprocess

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from champs_v4.fdtd_yee_3d import Yee3D
from champs_v4.visualization.vector_field_viz import VectorFieldVisualizer


def main():
    print("=" * 70)
    print(" Animation vectorielle - Refraction du champ E ".center(70))
    print("=" * 70)
    
    # Configuration
    nx, ny, nz = 180, 180, 1
    dx = 0.6e-3
    c0 = 3e8
    dt = 0.45 * dx / (c0 * np.sqrt(2))
    
    print(f"\nGrille : {nx}x{ny}x{nz}, dx = {dx*1e3:.2f} mm")
    
    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=12)
    
    # Interface air/dielectrique
    epsilon_r = np.ones((nx, ny, nz))
    epsilon_r[nx//2:, :, :] = 6.0  # Dielectrique epsilon_r = 6
    sigma = np.zeros((nx, ny, nz))
    sim.set_materials(epsilon_r, sigma)
    
    print(f"Interface : air (epsilon_r=1) | dielectrique (epsilon_r=6)")
    
    # Source
    source_x = 35
    freq = 12e9
    omega = 2 * np.pi * freq
    
    print(f"Source : onde plane a {freq/1e9:.0f} GHz")
    
    # Parametres animation
    nsteps = 600
    frame_interval = 6
    
    out_dir = parent_dir / 'champs_v4' / 'results' / 'anim_vector_field'
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSimulation de {nsteps} pas, {nsteps//frame_interval} frames...")
    
    # Creer visualiseur avec auto-detection
    print("\nInitialisation du visualiseur...")
    viz = VectorFieldVisualizer(sim, field='auto', z_index=0)
    
    # Enveloppe gaussienne
    t0 = 120 * dt
    width = 50 * dt
    
    frame_count = 0
    
    # Choisir mode de visualisation : 'streamlines', 'quiver', ou 'hybrid'
    viz_mode = 'hybrid'  # Changez ici pour tester les differents modes
    
    print(f"Mode de visualisation : {viz_mode.upper()}")
    
    for n in range(nsteps):
        t = n * dt
        envelope = np.exp(-0.5 * ((t - t0) / width) ** 2)
        source_value = envelope * np.sin(omega * t)
        
        # Injection onde plane
        for j in range(ny):
            sim.Ez[source_x, j, 0] += source_value * 0.5
        
        sim.step()
        
        # Generer frame
        if n % frame_interval == 0:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Choisir le mode de visualisation
            if viz_mode == 'streamlines':
                viz.plot_streamlines(ax, density=1.8, color_by_magnitude=True)
            elif viz_mode == 'quiver':
                viz.plot_quiver(ax, step=5, scale=25, show_magnitude_bg=True)
            else:  # hybrid
                viz.plot_hybrid(ax, streamline_density=1.5, quiver_step=9)
            
            # Marquer l'interface
            ax.axvline(x=nx//2, color='yellow', linestyle='--', 
                      linewidth=2.5, alpha=0.7, label='Interface')
            
            # Info temporelle
            ax.text(0.02, 0.98, f'Pas : {n}/{nsteps}\\nt = {t*1e12:.1f} ps',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.legend(loc='upper right')
            ax.set_title(f'Champ E vectoriel - Refraction (mode: {viz_mode})', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(frames_dir / f'frame_{frame_count:04d}.png', dpi=100)
            plt.close(fig)
            
            frame_count += 1
            
            if n % 60 == 0:
                print(f"  Pas {n}/{nsteps} - {frame_count} frames")
    
    print(f"\nFrames generees : {frame_count}")
    
    # Creer MP4
    mp4_path = out_dir / f'refraction_vector_{viz_mode}.mp4'
    print(f"\nCreation de l'animation MP4...")
    
    try:
        cmd = [
            'ffmpeg', '-y', '-framerate', '25',
            '-i', str(frames_dir / 'frame_%04d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '20', str(mp4_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"OK Animation creee : {mp4_path}")
    except subprocess.CalledProcessError:
        print(f"ATT FFmpeg erreur, frames dans : {frames_dir}")
    except FileNotFoundError:
        print(f"ATT FFmpeg non trouve, frames dans : {frames_dir}")
    
    print(f"\n{'='*70}")
    print(f"Animation terminee !")
    print(f"  - Frames PNG : {frames_dir}")
    print(f"  - Video MP4  : {mp4_path}")
    print(f"  - Mode       : {viz_mode}")
    print(f"\nPour essayer un autre mode, modifiez 'viz_mode' dans le script :")
    print(f"  - 'streamlines' : lignes de champ continues")
    print(f"  - 'quiver'      : fleches vectorielles")
    print(f"  - 'hybrid'      : combinaison des deux")


if __name__ == '__main__':
    main()
