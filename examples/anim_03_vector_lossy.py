"""
Animation 3 VECTORIELLE : Attenuation dans un milieu avec pertes
Montre les VECTEURS du champ s'attenuant progressivement.
"""
import numpy as np
import sys
import os
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from champs_v4.fdtd_yee_3d import Yee3D
from champs_v4.visualization.vector_field_viz import VectorFieldVisualizer
import matplotlib.pyplot as plt
import subprocess


def main():
    print("=" * 70)
    print(" " * 10 + "Animation 3 : ATTENUATION - Vecteurs du champ")
    print("=" * 70)
    print()
    
    # Grille 2D
    nx, ny, nz = 220, 180, 1
    dx = 0.6e-3  # 0.6 mm
    c0 = 3e8
    dt = 0.45 * dx / (c0 * np.sqrt(2))
    
    print(f"Configuration :")
    print(f"  Grille : {nx}x{ny}x{nz}")
    print(f"  Resolution : dx = {dx*1e3:.2f} mm")
    print()
    
    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=12)
    
    # Milieu avec pertes (conductivite faible)
    transition_x = nx // 3
    sim.sigma[transition_x:, :, :] = 5.0  # Faible conductivite
    sim.epsilon_r[transition_x:, :, :] = 2.5  # Dielectrique
    
    print(f"Materiaux :")
    print(f"  Gauche (x < {transition_x}) : AIR (sigma = 0)")
    print(f"  Droite (x > {transition_x}) : MILIEU ABSORBANT")
    print(f"    - epsilon_r = 2.5")
    print(f"    - sigma = 5.0 S/m (pertes)")
    print()
    
    # Source
    source_x = 25
    source_y = ny // 2
    freq = 6e9  # 6 GHz
    omega = 2 * np.pi * freq
    
    print(f"Source :")
    print(f"  Position : x={source_x}, y={source_y}")
    print(f"  Type : Impulsion gaussienne")
    print(f"  Frequence : {freq/1e9:.1f} GHz")
    print()
    
    # Parametres impulsion
    t0 = 100 * dt
    spread = 35 * dt
    
    # Animation
    nsteps = 1200
    frame_interval = 6
    
    print(f"Animation : {nsteps} pas, {nsteps//frame_interval} frames")
    print()
    
    # Dossier sortie
    out_dir = parent_dir / 'champs_v4' / 'results' / 'anim_03_vectors'
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    print("Simulation et generation des frames...")
    frame_count = 0
    
    for n in range(nsteps):
        # Source impulsionnelle
        t = n * dt
        pulse = np.exp(-((t - t0) / spread)**2) * np.sin(omega * t)
        sim.Ez[source_x, source_y, 0] += pulse
        
        sim.step()
        
        # Generer frame
        if n % frame_interval == 0:
            if frame_count % 20 == 0:
                print(f"  Frame {frame_count}/{nsteps//frame_interval} (pas {n}/{nsteps})")
            
            # Creer visualiseur avec donnees actuelles
            viz = VectorFieldVisualizer(sim, field='auto', z_index=0)
            
            fig, axes = plt.subplots(1, 2, figsize=(17, 7))
            
            # Vue 1 : Vecteurs normalises (voir attenuation direction)
            viz.plot_normalized(axes[0], step=7, arrow_scale=3.5,
                               show_magnitude_bg=True, cmap='hot')
            axes[0].axvline(transition_x, color='cyan', linestyle='--',
                           linewidth=2, alpha=0.8, label='Transition')
            axes[0].legend(loc='upper right', fontsize=9)
            axes[0].set_title('Vecteurs normalises', fontsize=11)
            axes[0].text(transition_x + 20, 10, 'MILIEU ABSORBANT',
                        color='cyan', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            # Vue 2 : Streamlines (voir attenuation flux)
            viz.plot_streamlines(axes[1], density=1.0, color_by_magnitude=True)
            axes[1].axvline(transition_x, color='yellow', linestyle='--',
                           linewidth=2, alpha=0.8)
            axes[1].set_title('Lignes de champ', fontsize=11)
            
            fig.suptitle(f'Attenuation dans milieu avec pertes - t = {t*1e12:.1f} ps',
                        fontsize=13, fontweight='bold')
            plt.tight_layout()
            
            frame_path = frames_dir / f'frame_{frame_count:04d}.png'
            plt.savefig(frame_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            
            frame_count += 1
    
    print()
    print(f"Total frames : {frame_count}")
    print()
    
    # Video
    video_path = out_dir / 'attenuation_vectors.mp4'
    print(f"Generation video : {video_path.name}")
    
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-framerate', '30',
        '-i', str(frames_dir / 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
        str(video_path)
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"Video : {video_path}")
    except:
        print(f"Frames dans : {frames_dir}")
    
    print()
    print("=" * 70)
    print("RESULTAT :")
    print("  - Onde se propage normalement dans l'air")
    print("  - Attenuation EXPONENTIELLE dans le milieu avec pertes")
    print("  - Amplitude diminue progressivement")
    print("  - Energie convertie en chaleur (pertes Joule)")
    print("  - Vecteurs de plus en plus faibles")
    print("=" * 70)


if __name__ == '__main__':
    main()
