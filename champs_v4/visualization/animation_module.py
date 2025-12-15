"""
Module d'animation pour le projet FDTD Yee3D.

Ce module permet de créer des animations (MP4) à partir d'un ensemble de frames PNG
générés par une simulation, ou de générer des animations 3D directement.

Fonctions:
- create_animation(frames_dir, output_file, framerate=10, mode='slice', **kwargs)

Paramètres:
- frames_dir: Dossier contenant les frames PNG (pour mode 'slice')
- output_file: Chemin du fichier MP4 de sortie
- framerate: Images par seconde (défaut: 10)
- mode: 'slice' pour coupes 2D, '3d' pour animations 3D générées
- **kwargs: Options supplémentaires (sim, field, etc. pour 3D)
"""

import os
import subprocess
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional

def create_animation(
    frames_dir: str,
    output_file: str,
    frame_rate: int = 10,
    mode: str = 'slice',
    crf: int = 23,
    prefix: str = 'frame',
    **kwargs
) -> Optional[str]:
    """
    Crée une animation MP4 à partir des frames PNG ou génère une animation 3D.

    Args:
        frames_dir: Dossier des frames (pour 'slice') ou où sauvegarder les frames 3D
        output_file: Fichier MP4 de sortie
        frame_rate: FPS de l'animation
        mode: 'slice' pour frames pré-générés, '3d' pour génération 3D
        crf: Qualité de compression (0-51, défaut 23)
        prefix: Préfixe des noms de frames (pour 'slice')
        **kwargs: Options (pour 3D: sim, field='B', nsteps=100, frame_interval=5, sx=8, sy=8, sz=4, source_type='antenna', omega=2*pi*1e9, feed_pos=(nx//2,ny//2,nz//2), envelope_type='gaussian', t0=nsteps*dt/2, width=t0/4)

    Returns:
        Chemin du fichier MP4 créé, ou None si échec
    """
    if mode == '3d':
        return _create_3d_animation(frames_dir, output_file, frame_rate, crf, **kwargs)
        # return _create_slice_animation(frames_dir, output_file, frame_rate, crf, prefix, **kwargs)

def _create_3d_animation(
    frames_dir: str,
    output_file: str,
    frame_rate: int,
    crf: int,
    **kwargs
) -> Optional[str]:
    """
    Génère et crée une animation 3D à partir d'un objet Yee3D.
    """
    sim = kwargs.get('sim')
    if sim is None:
        print("Erreur: Pour mode='3d', fournir sim=Yee3D_instance dans kwargs")
        return None

    field = kwargs.get('field', 'B')
    nsteps = kwargs.get('nsteps', 100)
    frame_interval = kwargs.get('frame_interval', 5)
    sx = kwargs.get('sx', 8)  # downsample x
    sy = kwargs.get('sy', 8)  # downsample y
    sz = kwargs.get('sz', 4)  # downsample z

    # Source parameters
    source_type = kwargs.get('source_type', None)
    if source_type == 'antenna':
        omega = kwargs.get('omega', 2*np.pi*1e9)
        feed_pos = kwargs.get('feed_pos', (sim.nx//2, sim.ny//2, sim.nz//2))
        amplitude = kwargs.get('amplitude', 1.0)
        envelope_type = kwargs.get('envelope_type', 'gaussian')
        if envelope_type == 'gaussian':
            t0 = kwargs.get('t0', nsteps * sim.dt / 2)  # Milieu de la simulation
            width = kwargs.get('width', t0 / 4)  # Largeur 1/4 de la durée

    os.makedirs(frames_dir, exist_ok=True)

    # Grille d'échantillonnage
    xs = np.arange(0, sim.nx, sx)
    ys = np.arange(0, sim.ny, sy)
    zs = np.arange(0, sim.nz, sz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    frame_idx = 0
    for n in range(nsteps):
        # Inject source if antenna
        if source_type == 'antenna':
            t = n * sim.dt
            base_src = np.sin(omega * t)
            if envelope_type == 'gaussian':
                envelope = np.exp(-0.5 * ((t - t0) / width)**2)
                src = amplitude * envelope * base_src
            else:
                src = amplitude * base_src
            sim.Ez[feed_pos] += src

        sim.step()

        # Check for NaN
        if np.any(np.isnan(sim.Ex)) or np.any(np.isnan(sim.Ey)) or np.any(np.isnan(sim.Ez)):
            print(f"NaN detected in E fields at step {n}, stopping simulation")
            break

        if n % frame_interval == 0:
            # Calculer le champ
            if field.upper() == 'B':
                # B = mu0 * H, mais pour visualisation, on peut utiliser H directement
                U = sim.Hx[X, Y, Z]
                V = sim.Hy[X, Y, Z]
                W = sim.Hz[X, Y, Z]
            else:  # E
                U = sim.Ex[X, Y, Z]
                V = sim.Ey[X, Y, Z]
                W = sim.Ez[X, Y, Z]

            # Plot 3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.quiver(X, Y, Z, U, V, W, length=0.5, normalize=True, color='b', alpha=0.6)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{field}-field 3D at step {n}')
            ax.set_xlim(0, sim.nx)
            ax.set_ylim(0, sim.ny)
            ax.set_zlim(0, sim.nz)

            # Sauvegarder frame
            frame_path = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
            fig.savefig(frame_path, dpi=100)
            plt.close(fig)
            frame_idx += 1

    print(f"Généré {frame_idx} frames 3D.")

    # Maintenant créer l'animation avec les frames générés
    return _create_slice_animation(frames_dir, output_file, frame_rate, crf, 'frame')
def _create_slice_animation(
    frames_dir: str,
    output_file: str,
    framerate: int,
    crf: int,
    prefix: str,
    **kwargs
) -> Optional[str]:
    """
    Crée une animation MP4 à partir des frames PNG pré-générés.
    """
    if not os.path.exists(frames_dir):
        print(f"Erreur: Dossier {frames_dir} n'existe pas.")
        return None

    # Trouver tous les frames PNG
    frame_pattern = os.path.join(frames_dir, f'{prefix}_*.png')
    frames = sorted(glob.glob(frame_pattern))

    if not frames:
        print(f"Aucun frame trouvé dans {frames_dir} avec préfixe '{prefix}'.")
        return None

    print(f"Trouvé {len(frames)} frames.")

    # Vérifier si les noms sont consécutifs
    expected_names = [f"{prefix}_{i:04d}.png" for i in range(len(frames))]
    actual_names = [os.path.basename(f) for f in frames]

    if actual_names != expected_names:
        print("Renommage des frames pour séquence consécutive...")
        temp_frames = []
        for i, old_path in enumerate(frames):
            new_name = f"{prefix}_{i:04d}.png"
            new_path = os.path.join(frames_dir, new_name)
            os.rename(old_path, new_path)
            temp_frames.append(new_path)
        frames = temp_frames

    # Créer le dossier de sortie si nécessaire
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Commande FFmpeg
    input_pattern = os.path.join(frames_dir, f'{prefix}_%04d.png')
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(framerate),
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', str(crf),
        '-preset', 'medium',
        output_file
    ]

    # Ajouter options supplémentaires
    for key, value in kwargs.items():
        if key.startswith('ffmpeg_'):
            cmd.extend([f'-{key[7:]}', str(value)])
        else:
            cmd.extend([f'-{key}', str(value)])

    print(f"Exécution de FFmpeg: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Animation créée avec succès:", output_file)
        print(f"Durée: {len(frames)/framerate:.1f}s, {len(frames)} frames à {framerate} fps")
        return output_file
    except subprocess.CalledProcessError as e:
        print("Erreur FFmpeg:")
        print(e.stderr)
        return None
    except FileNotFoundError:
        print("FFmpeg n'est pas installé ou pas dans le PATH.")
        print("Installez FFmpeg depuis https://ffmpeg.org/download.html")
        return None

# Exemple d'utilisation
if __name__ == '__main__':
    # Exemple pour une simulation d'antenne
    frames_dir = 'results/antenna_animation/frames'
    output_file = 'results/antenna_animation/antenna_animation.mp4'
    create_animation(frames_dir, output_file, frame_rate=15, crf=20)
