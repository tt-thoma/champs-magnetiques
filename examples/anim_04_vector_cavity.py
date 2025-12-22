"""
Animation 4 VECTORIELLE : Resonance dans une cavite dielectrique
Montre les VECTEURS formant des modes resonnants.
"""
import numpy as np
import sys
import os
from pathlib import Path

from champs_v4.fdtd_yee_3d import Yee3D
from champs_v4.visualization.vector_field_viz import VectorFieldVisualizer
import matplotlib.pyplot as plt
import subprocess


def main():
    print("=" * 70)
    print(" " * 10 + "Animation 4 : CAVITE RESONANTE - Vecteurs du champ")
    print("=" * 70)
    print()
    
    # Grille 2D
    nx, ny, nz = 160, 160, 1
    dx = 0.7e-3  # 0.7 mm
    c0 = 3e8
    dt = 0.45 * dx / (c0 * np.sqrt(2))
    
    print(f"Configuration :")
    print(f"  Grille : {nx}x{ny}x{nz}")
    print(f"  Resolution : dx = {dx*1e3:.2f} mm")
    print()
    
    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=10)
    
    # Cavite rectangulaire au centre
    cx, cy = nx // 2, ny // 2
    width, height = 60, 60
    
    x1, x2 = cx - width//2, cx + width//2
    y1, y2 = cy - height//2, cy + height//2
    
    # Remplir cavite avec dielectrique
    sim.epsilon_r[x1:x2, y1:y2, :] = 4.0  # epsilon_r = 4 (n=2)
    
    # Murs conducteurs (bordures de la cavite)
    wall_sigma = 1e7
    thickness = 3
    # Murs verticaux
    sim.sigma[x1:x1+thickness, y1:y2, :] = wall_sigma
    sim.sigma[x2-thickness:x2, y1:y2, :] = wall_sigma
    # Murs horizontaux
    sim.sigma[x1:x2, y1:y1+thickness, :] = wall_sigma
    sim.sigma[x1:x2, y2-thickness:y2, :] = wall_sigma
    
    print(f"Materiaux :")
    print(f"  CAVITE rectangulaire : {width}x{height} cellules")
    print(f"  Centre : ({cx}, {cy})")
    print(f"  Interieur : epsilon_r = 4.0 (n = 2.0)")
    print(f"  Murs : conducteurs (sigma = {wall_sigma:.0e} S/m)")
    print()
    
    # Source au centre de la cavite
    source_x = cx
    source_y = cy
    freq = 8e9  # 8 GHz
    omega = 2 * np.pi * freq
    
    print(f"Source :")
    print(f"  Position : centre cavite ({source_x}, {source_y})")
    print(f"  Type : Impulsion gaussienne")
    print(f"  Frequence : {freq/1e9:.1f} GHz")
    print()
    
    # Parametres impulsion
    t0 = 60 * dt
    spread = 20 * dt
    
    # Animation
    nsteps = 1500
    frame_interval = 8
    
    print(f"Animation : {nsteps} pas, {nsteps//frame_interval} frames")
    print()
    
    # Dossier sortie
    out_dir = parent_dir / 'champs_v4' / 'results' / 'anim_04_vectors'
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    print("Simulation et generation des frames...")
    frame_count = 0
    
    for n in range(nsteps):
        # Source impulsionnelle courte
        t = n * dt
        pulse = np.exp(-((t - t0) / spread)**2) * np.sin(omega * t)
        sim.Ez[source_x, source_y, 0] += pulse
        
        sim.step()
        
        # Generer frame
        if n % frame_interval == 0:
            if frame_count % 15 == 0:
                print(f"  Frame {frame_count}/{nsteps//frame_interval} (pas {n}/{nsteps})")
            
            # Creer visualiseur avec donnees actuelles
            viz = VectorFieldVisualizer(sim, field='auto', z_index=0)            
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            # Vue 1 : Vecteurs normalises (modes clairs)
            viz.plot_normalized(axes[0], step=5, arrow_scale=3.0,
                               show_magnitude_bg=True, cmap='viridis')
            # Dessiner contour cavite
            from matplotlib.patches import Rectangle
            rect = Rectangle((x1, y1), width, height, 
                            fill=False, edgecolor='yellow', 
                            linewidth=3, linestyle='-')
            axes[0].add_patch(rect)
            axes[0].text(cx, y2 + 8, 'CAVITE RESONANTE',
                        color='yellow', fontsize=11, fontweight='bold',
                        ha='center', bbox=dict(boxstyle='round',
                        facecolor='black', alpha=0.7))
            axes[0].set_title('Vecteurs normalises', fontsize=11)
            
            # Vue 2 : Streamlines (voir modes resonnants)
            viz.plot_streamlines(axes[1], density=1.2, color_by_magnitude=True)
            rect2 = Rectangle((x1, y1), width, height,
                             fill=False, edgecolor='cyan',
                             linewidth=3, linestyle='-')
            axes[1].add_patch(rect2)
            axes[1].set_title('Lignes de champ', fontsize=11)
            
            fig.suptitle(f'Modes resonnants dans cavite - t = {t*1e12:.1f} ps',
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
    video_path = out_dir / 'cavite_resonante_vectors.mp4'
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
    print("  - Onde initiale excite la cavite")
    print("  - Formation de MODES RESONNANTS")
    print("  - Patterns stationnaires complexes")
    print("  - Energie confinee dans la cavite")
    print("  - Vecteurs montrent structure modale")
    print("=" * 70)


if __name__ == '__main__':
    main()
