"""
Animation 2 VECTORIELLE : Reflexion sur un conducteur metallique
Montre les VECTEURS du champ se reflechissant sur une plaque metallique.
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
    print(" " * 10 + "Animation 2 : REFLEXION METAL - Vecteurs du champ")
    print("=" * 70)
    print()
    
    # Grille 2D
    nx, ny, nz = 180, 180, 1
    dx = 0.8e-3  # 0.8 mm
    c0 = 3e8
    dt = 0.45 * dx / (c0 * np.sqrt(2))
    
    print(f"Configuration :")
    print(f"  Grille : {nx}x{ny}x{nz}")
    print(f"  Resolution : dx = {dx*1e3:.2f} mm")
    print()
    
    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=12)
    
    # Plaque metallique verticale
    plate_x = nx // 2
    thickness = 5
    sim.sigma[plate_x:plate_x+thickness, :, :] = 5.8e7  # Cuivre
    
    print(f"Materiaux :")
    print(f"  Air + Plaque CUIVRE verticale")
    print(f"  Position plaque : x = {plate_x}")
    print(f"  Epaisseur : {thickness} cellules")
    print(f"  Conductivite : 5.8e7 S/m")
    print()
    
    # Source ponctuelle
    source_x = 30
    source_y = ny // 2
    freq = 5e9  # 5 GHz
    omega = 2 * np.pi * freq
    
    print(f"Source :")
    print(f"  Position : x={source_x}, y={source_y}")
    print(f"  Type : Impulsion gaussienne")
    print(f"  Frequence : {freq/1e9:.1f} GHz")
    print()
    
    # Parametres impulsion
    t0 = 120 * dt
    spread = 40 * dt
    
    # Animation
    nsteps = 1000
    frame_interval = 5
    
    print(f"Animation : {nsteps} pas, {nsteps//frame_interval} frames")
    print()
    
    # Dossier sortie
    out_dir = parent_dir / 'champs_v4' / 'results' / 'anim_02_vectors'
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
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            # Vue 1 : Vecteurs normalises
            viz.plot_normalized(axes[0], step=5, arrow_scale=3.5,
                               show_magnitude_bg=True, cmap='plasma')
            # Dessiner plaque
            for i in range(thickness):
                axes[0].axvline(plate_x + i, color='yellow', 
                               linewidth=3, alpha=0.8)
            axes[0].text(plate_x + thickness//2, ny - 10, 'METAL', 
                        color='yellow', fontsize=12, fontweight='bold',
                        ha='center', bbox=dict(boxstyle='round', 
                        facecolor='black', alpha=0.7))
            axes[0].set_title('Vecteurs normalises', fontsize=11)
            
            # Vue 2 : Hybride
            viz.plot_hybrid(axes[1], streamline_density=0.7, quiver_step=12)
            for i in range(thickness):
                axes[1].axvline(plate_x + i, color='yellow', 
                               linewidth=3, alpha=0.8)
            axes[1].set_title('Vue hybride', fontsize=11)
            
            fig.suptitle(f'Reflexion sur conducteur - t = {t*1e12:.1f} ps',
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
    video_path = out_dir / 'reflexion_metal_vectors.mp4'
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
    print("  - Onde spherique emise depuis la source")
    print("  - Reflexion TOTALE sur le conducteur")
    print("  - Onde stationnaire formee (interference)")
    print("  - Champ nul a l'interieur du metal")
    print("  - Vecteurs inverses apres reflexion")
    print("=" * 70)


if __name__ == '__main__':
    main()
