"""
Animation 2 : Réflexion sur un conducteur métallique
Montre une onde se réfléchissant sur une plaque métallique (haute conductivité).
"""
import numpy as np
import sys
import os
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from champs_v4.fdtd_yee_3d import Yee3D
import matplotlib.pyplot as plt
import subprocess


def main():
    print("=" * 60)
    print("Animation 2 : Reflexion sur un conducteur metallique")
    print("=" * 60)
    
    # Grille 2D
    nx, ny, nz = 180, 180, 1
    dx = 0.8e-3  # 0.8 mm
    c0 = 3e8
    dt = 0.45 * dx / (c0 * np.sqrt(2))
    
    print(f"Grille: {nx}x{ny}x{nz}, dx={dx*1e3:.2f}mm")
    
    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=12)
    
    # Créer plaque métallique verticale au centre
    epsilon_r = np.ones((nx, ny, nz))
    sigma = np.zeros((nx, ny, nz))
    
    # Cuivre : sigma environ 5.8e7 S/m (tres conducteur)
    # Plaque de 5 cellules d'epaisseur
    plate_x = nx // 2
    thickness = 5
    sigma[plate_x:plate_x+thickness, :, :] = 5.8e7
    epsilon_r[plate_x:plate_x+thickness, :, :] = 1.0  # epsilon_r du cuivre environ 1
    
    sim.set_materials(epsilon_r, sigma)
    
    print(f"Materiaux :")
    print(f"  - Air : sigma = 0")
    print(f"  - Plaque metallique (cuivre) : sigma = 5.8e7 S/m")
    print(f"  - Position : x = {plate_x}, epaisseur = {thickness} cellules")
    
    # Source ponctuelle : impulsion gaussienne
    source_pos = (30, ny//2, 0)
    freq = 5e9  # 5 GHz
    omega = 2 * np.pi * freq
    
    print(f"Source : Impulsion gaussienne a {source_pos[:2]}")
    print(f"Frequence : {freq/1e9:.1f} GHz")
    
    nsteps = 1000
    frame_interval = 5
    
    # Créer dossier frames
    out_dir = parent_dir / 'champs_v4' / 'results' / 'anim_02_metal'
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Parametres impulsion gaussienne
    t0 = 120 * dt
    spread = 40 * dt
    
    print(f"Simulation de {nsteps} pas...")
    
    frame_count = 0
    for n in range(nsteps):
        t = n * dt
        pulse = np.exp(-((t - t0) / spread)**2) * np.sin(omega * t)
        source_value = 0.5 * pulse
        
        sim.Ez[source_pos] += source_value
        sim.step()
        
        if n % frame_interval == 0:
            Ez_slice = sim.Ez[:, :, 0]
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(Ez_slice.T, origin='lower', cmap='seismic', 
                          vmin=-0.3, vmax=0.3, extent=[0, nx, 0, ny])
            ax.axvline(x=plate_x, color='gold', linewidth=3, label='Plaque métallique')
            ax.set_title(f'Réflexion sur conducteur - Pas {n}/{nsteps}')
            ax.legend()
            plt.colorbar(im, ax=ax, label='Ez (V/m)')
            plt.tight_layout()
            plt.savefig(frames_dir / f'frame_{frame_count:04d}.png', dpi=100)
            plt.close(fig)
            frame_count += 1
        
        if n % 150 == 0:
            print(f"  Pas {n}/{nsteps} - {frame_count} frames")
    
    mp4_path = out_dir / 'metal_reflection.mp4'
    try:
        subprocess.run(['ffmpeg', '-y', '-framerate', '20', '-i', str(frames_dir / 'frame_%04d.png'),
                       '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', str(mp4_path)],
                      check=True, capture_output=True)
        print(f"\nOK Animation creee : {mp4_path}")
    except:
        print(f"\nATT Frames dans : {frames_dir}")
    
    print(f"OK Animation sauvegardee dans : {out_dir}")
    print("  Phenomene : Reflexion quasi-totale + absorption dans le metal")


if __name__ == '__main__':
    main()
