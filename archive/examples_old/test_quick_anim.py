"""
Test rapide d'animation (version accélérée pour tester que tout fonctionne).
"""
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from champs_v4.fdtd_yee_3d import Yee3D


def main():
    print("Test rapide d'animation FDTD")
    print("=" * 50)
    
    # Grille TRÈS petite pour test rapide
    nx, ny, nz = 80, 80, 1
    dx = 1e-3  # 1 mm
    c0 = 3e8
    dt = 0.45 * dx / (c0 * np.sqrt(2))
    
    print(f"Grille: {nx}x{ny}x{nz}")
    
    sim = Yee3D(nx, ny, nz, dx, dt, pml_width=8)
    
    # Interface simple
    epsilon_r = np.ones((nx, ny, nz))
    epsilon_r[nx//2:, :, :] = 4.0  # Diélectrique simple
    sigma = np.zeros((nx, ny, nz))
    sim.set_materials(epsilon_r, sigma)
    
    print("Interface air/diélectrique configurée")
    
    # Source
    source_pos = (20, ny//2, 0)
    freq = 5e9
    omega = 2 * np.pi * freq
    
    # Seulement 100 pas pour test rapide
    nsteps = 100
    frame_interval = 10  # Seulement 10 frames
    
    print(f"Simulation de {nsteps} pas (test rapide)...")
    
    out_dir = parent_dir / 'champs_v4' / 'results' / 'test_quick'
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    t0 = 30 * dt
    width = 15 * dt
    
    frame_count = 0
    for n in range(nsteps):
        t = n * dt
        envelope = np.exp(-0.5 * ((t - t0) / width) ** 2)
        source_value = 0.5 * envelope * np.sin(omega * t)
        
        sim.Ez[source_pos] += source_value
        sim.step()
        
        if n % frame_interval == 0:
            Ez_slice = sim.Ez[:, :, 0]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(Ez_slice.T, origin='lower', cmap='RdBu_r', 
                          vmin=-0.3, vmax=0.3)
            ax.axvline(x=nx//2, color='yellow', linestyle='--', linewidth=2)
            ax.set_title(f'Test - Pas {n}/{nsteps}')
            plt.colorbar(im, ax=ax, label='Ez')
            plt.tight_layout()
            
            plt.savefig(frames_dir / f'frame_{frame_count:04d}.png', dpi=80)
            plt.close(fig)
            frame_count += 1
            print(f"  Frame {frame_count}/{nsteps//frame_interval}")
    
    print(f"\n✓ Test terminé !")
    print(f"  {frame_count} frames générées dans : {frames_dir}")
    print(f"\nSi vous voyez ce message, les animations devraient fonctionner !")
    print(f"Lancez maintenant : python examples/anim_01_dielectric_refraction.py")


if __name__ == '__main__':
    main()
