"""
Animation 3 : Propagation dans un milieu avec pertes (eau salée)
Montre l'atténuation d'une onde dans un milieu conducteur.
"""
import numpy as np
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from champs_v4.fdtd_yee_3d import Yee3D
import matplotlib.pyplot as plt
import subprocess


def main():
    print("=" * 60)
    print("Animation 3 : Propagation dans milieu avec pertes")
    print("=" * 60)
    
    # Grille 2D
    nx, ny, nz = 220, 220, 1
    dx = 0.6e-3
    c0 = 3e8
    dt = 0.45 * dx / (c0 * np.sqrt(2))
    
    print(f"Grille: {nx}x{ny}x{nz}, dx={dx*1e3:.2f}mm")
    
    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=15)
    
    # Milieu : eau salee (conductivite moderee)
    epsilon_r = np.ones((nx, ny, nz))
    sigma = np.zeros((nx, ny, nz))
    
    # Air dans premiere partie, eau salee dans le reste
    transition_x = 50
    
    # Eau : epsilon_r environ 80, eau salee : sigma environ 4 S/m
    epsilon_r[transition_x:, :, :] = 80.0
    sigma[transition_x:, :, :] = 4.0
    
    sim.set_materials(epsilon_r, sigma)
    
    print(f"Materiaux :")
    print(f"  - Air (x < {transition_x}) : epsilon_r = 1, sigma = 0")
    print(f"  - Eau salee (x >= {transition_x}) : epsilon_r = 80, sigma = 4 S/m")
    
    # Source ponctuelle dans l'air
    source_pos = (20, ny//2, 0)
    freq = 6e9  # 6 GHz
    omega = 2 * np.pi * freq
    
    print(f"Source : Impulsion gaussienne")
    print(f"Frequence : {freq/1e9:.1f} GHz")
    
    nsteps = 1200
    frame_interval = 6
    
    out_dir = parent_dir / 'champs_v4' / 'results' / 'anim_03_lossy'
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Parametres impulsion gaussienne
    t0 = 100 * dt
    spread = 35 * dt
    
    print(f"Simulation de {nsteps} pas (atténuation progressive)...")
    
    frame_count = 0
    for n in range(nsteps):
        t = n * dt
        pulse = np.exp(-((t - t0) / spread)**2) * np.sin(omega * t)
        source_value = pulse
        
        sim.Ez[source_pos] += source_value
        sim.step()
        
        if np.any(np.isnan(sim.Ez)):
            print(f"  ⚠ NaN détecté au pas {n}, arrêt")
            break
        
        if n % frame_interval == 0:
            Ez_slice = sim.Ez[:, :, 0]
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(Ez_slice.T, origin='lower', cmap='plasma', 
                          vmin=-0.2, vmax=0.2, extent=[0, nx, 0, ny])
            ax.axvline(x=transition_x, color='cyan', linestyle='--', linewidth=2, label='Interface eau')
            ax.set_title(f'Atténuation dans eau salée - Pas {n}/{nsteps}')
            ax.legend()
            plt.colorbar(im, ax=ax, label='Ez (V/m)')
            plt.tight_layout()
            plt.savefig(frames_dir / f'frame_{frame_count:04d}.png', dpi=100)
            plt.close(fig)
            frame_count += 1
        
        if n % 200 == 0:
            print(f"  Pas {n}/{nsteps} - {frame_count} frames")
    
    mp4_path = out_dir / 'lossy_medium.mp4'
    try:
        subprocess.run(['ffmpeg', '-y', '-framerate', '20', '-i', str(frames_dir / 'frame_%04d.png'),
                       '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', str(mp4_path)],
                      check=True, capture_output=True)
        print(f"\nOK Animation creee : {mp4_path}")
    except:
        print(f"\nATT Frames dans : {frames_dir}")
    
    print(f"OK Animation sauvegardee dans : {out_dir}")
    print("  Phenomene : Attenuation exponentielle dans le milieu conducteur")


if __name__ == '__main__':
    main()
