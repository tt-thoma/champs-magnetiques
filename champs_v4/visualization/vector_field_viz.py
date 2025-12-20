"""
Module de visualisation vectorielle des champs EM.

Ce module propose plusieurs representations vectorielles des champs :
1. Streamlines (lignes de champ)
2. Quiver (fleches)
3. Hybride (magnitude + vecteurs)
4. Normalise (vecteurs de meme longueur)
5. LIC (Line Integral Convolution)

IMPORTANT pour simulations 2D :
- Si Ez domine (mode TM) : visualise les vecteurs H dans le plan (Hx, Hy)
- Si Hz domine (mode TE) : visualise les vecteurs E dans le plan (Ex, Ey)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from typing import Optional, Tuple, Literal


def _centered_E_plane(sim, k):
    """Extraire Ex, Ey sur les centres de cellules pour z=k."""
    nx, ny = sim.nx, sim.ny
    Ex_c = 0.5 * (sim.Ex[:, 0:ny, k] + sim.Ex[:, 1:ny + 1, k])
    Ey_c = 0.5 * (sim.Ey[0:nx, :, k] + sim.Ey[1:nx + 1, :, k])
    return Ex_c, Ey_c


def _centered_H_plane(sim, k):
    """Extraire Hx, Hy sur les centres de cellules pour z=k."""
    nx, ny = sim.nx, sim.ny
    Hx_c = 0.5 * (sim.Hx[0:nx, :, k] + sim.Hx[1:nx + 1, :, k])
    Hy_c = 0.5 * (sim.Hy[:, 0:ny, k] + sim.Hy[:, 1:ny + 1, k])
    return Hx_c, Hy_c


def _detect_dominant_mode(sim, k=0):
    """
    Detecter le mode dominant (TM ou TE) en comparant les magnitudes.
    
    Returns:
        'TM' si Ez domine (alors visualiser H dans le plan)
        'TE' si Hz domine (alors visualiser E dans le plan)
        'MIXED' si les deux sont significatifs
    """
    # Calculer magnitudes moyennes
    Ez_mag = np.abs(sim.Ez[:, :, k]).max()
    Hz_mag = np.abs(sim.Hz[:, :, k]).max()
    
    Ex_c, Ey_c = _centered_E_plane(sim, k)
    Exy_mag = np.sqrt(Ex_c**2 + Ey_c**2).max()
    
    Hx_c, Hy_c = _centered_H_plane(sim, k)
    Hxy_mag = np.sqrt(Hx_c**2 + Hy_c**2).max()
    
    # Si Ez >> Exy, c'est mode TM (visualiser H)
    if Ez_mag > 10 * Exy_mag and Hxy_mag > 1e-15:
        return 'TM'
    # Si Hz >> Hxy, c'est mode TE (visualiser E)
    elif Hz_mag > 10 * Hxy_mag and Exy_mag > 1e-15:
        return 'TE'
    else:
        return 'MIXED'


class VectorFieldVisualizer:
    """
    Classe pour visualiser les champs vectoriels EM.
    
    Modes disponibles :
    - 'streamlines' : Lignes de champ continues
    - 'quiver' : Fleches directionnelles
    - 'hybrid' : Fond (magnitude) + streamlines
    - 'lic' : Line Integral Convolution (texture)
    
    AUTO-DETECTION pour simulations 2D :
    - Si field='E' mais Ez domine : affiche automatiquement H (dans le plan)
    - Si field='H' mais Hz domine : affiche automatiquement E (dans le plan)
    """
    
    def __init__(self, sim, field: Literal['E', 'H', 'B', 'auto'] = 'auto', z_index: int = 0):
        """
        Initialiser le visualiseur.
        
        Args:
            sim : Instance Yee3D
            field : Type de champ ('E', 'H', 'B', ou 'auto' pour detection auto)
            z_index : Indice de la tranche z a visualiser
        """
        self.sim = sim
        self.z_index = z_index
        self.nx = sim.nx
        self.ny = sim.ny
        
        # Detection automatique du mode
        if field.lower() == 'auto':
            mode = _detect_dominant_mode(sim, z_index)
            if mode == 'TM':
                self.field = 'H'
                print(f"  [Auto-detection] Mode TM detecte (Ez domine) -> Visualisation de H dans le plan")
            elif mode == 'TE':
                self.field = 'E'
                print(f"  [Auto-detection] Mode TE detecte (Hz domine) -> Visualisation de E dans le plan")
            else:
                self.field = 'H'  # Par defaut
                print(f"  [Auto-detection] Mode mixte -> Visualisation de H dans le plan")
        else:
            self.field = field.upper()
            # Verification et suggestion
            mode = _detect_dominant_mode(sim, z_index)
            if mode == 'TM' and self.field == 'E':
                print(f"  ATTENTION : Ez domine mais vous visualisez E(xy) qui est faible!")
                print(f"  -> Suggestion : utilisez field='H' ou field='auto'")
            elif mode == 'TE' and self.field == 'H':
                print(f"  ATTENTION : Hz domine mais vous visualisez H(xy) qui est faible!")
                print(f"  -> Suggestion : utilisez field='E' ou field='auto'")
        
    def get_field_components(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extraire les composantes du champ dans le plan xy."""
        if self.field == 'E':
            return _centered_E_plane(self.sim, self.z_index)
        else:  # 'H' ou 'B'
            return _centered_H_plane(self.sim, self.z_index)
    
    def plot_streamlines(self, 
                        ax: Optional[plt.Axes] = None,
                        density: float = 0.8,
                        color_by_magnitude: bool = True,
                        downsample: int = 1) -> plt.Axes:
        """
        Tracer les lignes de champ (streamlines).
        
        Args:
            ax : Axes matplotlib (cree si None)
            density : Densite des lignes (float, defaut 1.5)
            color_by_magnitude : Colorer par magnitude (sinon noir)
            downsample : Facteur de sous-echantillonnage (1 = aucun)
        
        Returns:
            Axes matplotlib
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extraire champs
        U, V = self.get_field_components()
        
        # Sous-echantillonnage si demande
        if downsample > 1:
            U = U[::downsample, ::downsample]
            V = V[::downsample, ::downsample]
            nx, ny = U.shape
        else:
            nx, ny = self.nx, self.ny
        
        # Grille
        x = np.linspace(0, self.nx, nx)
        y = np.linspace(0, self.ny, ny)
        
        # Magnitude
        magnitude = np.sqrt(U**2 + V**2)
        
        # Fond : magnitude
        if color_by_magnitude:
            im = ax.imshow(magnitude.T, origin='lower', 
                          extent=[0, self.nx, 0, self.ny],
                          cmap='viridis', alpha=0.6, aspect='equal')
            plt.colorbar(im, ax=ax, label=f'|{self.field}| (a.u.)')
        
        # Streamlines
        try:
            if color_by_magnitude:
                # Couleur selon magnitude
                strm = ax.streamplot(x, y, U.T, V.T, 
                                    density=density,
                                    color=magnitude.T,
                                    cmap='plasma',
                                    linewidth=1.2,
                                    arrowsize=1.5,
                                    arrowstyle='->',
                                    minlength=0.2)
            else:
                # Couleur uniforme
                strm = ax.streamplot(x, y, U.T, V.T, 
                                    density=density,
                                    color='white',
                                    linewidth=1.2,
                                    arrowstyle='->',
                                    arrowsize=1.5)
        except Exception as e:
            # Fallback : quiver si streamplot echoue
            print(f"Streamplot failed: {e}, using quiver instead")
            X, Y = np.meshgrid(x, y, indexing='ij')
            ax.quiver(X, Y, U, V, magnitude, 
                     cmap='plasma', scale=50, alpha=0.8)
        
        ax.set_xlabel('X (cellules)')
        ax.set_ylabel('Y (cellules)')
        ax.set_title(f'Champ {self.field} - Lignes de champ')
        ax.set_xlim(0, self.nx)
        ax.set_ylim(0, self.ny)
        
        return ax
    
    def plot_quiver(self,
                   ax: Optional[plt.Axes] = None,
                   step: int = 6,
                   scale: float = None,
                   show_magnitude_bg: bool = True) -> plt.Axes:
        """
        Tracer les vecteurs du champ (quiver).
        
        Args:
            ax : Axes matplotlib
            step : Pas d'echantillonnage des fleches (1 = toutes, 2 = 1/2, etc.)
            scale : Facteur d'echelle des fleches (None = auto)
            show_magnitude_bg : Afficher fond de magnitude
        
        Returns:
            Axes matplotlib
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extraire champs
        U, V = self.get_field_components()
        
        # Sous-echantillonnage pour lisibilite
        U_sub = U[::step, ::step]
        V_sub = V[::step, ::step]
        
        # Grille
        x = np.arange(0, self.nx, step)
        y = np.arange(0, self.ny, step)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Magnitude
        magnitude = np.sqrt(U**2 + V**2)
        magnitude_sub = np.sqrt(U_sub**2 + V_sub**2)
        
        # Fond : magnitude complete
        if show_magnitude_bg:
            im = ax.imshow(magnitude.T, origin='lower',
                          extent=[0, self.nx, 0, self.ny],
                          cmap='gray', alpha=0.3, aspect='equal')
        
        # Fleches colorees par magnitude
        quiv = ax.quiver(X, Y, U_sub, V_sub, magnitude_sub,
                        cmap='jet', scale=scale, 
                        pivot='mid', alpha=0.85,
                        width=0.004, headwidth=3.5, headlength=4.5,
                        scale_units='xy')
        
        plt.colorbar(quiv, ax=ax, label=f'|{self.field}| (a.u.)')
        
        ax.set_xlabel('X (cellules)')
        ax.set_ylabel('Y (cellules)')
        ax.set_title(f'Champ {self.field} - Vecteurs')
        ax.set_xlim(0, self.nx)
        ax.set_ylim(0, self.ny)
        ax.set_aspect('equal')
        
        return ax
    
    def plot_hybrid(self,
                   ax: Optional[plt.Axes] = None,
                   streamline_density: float = 0.6,
                   quiver_step: int = 12) -> plt.Axes:
        """
        Visualisation hybride : fond (magnitude) + streamlines + quelques fleches.
        
        Args:
            ax : Axes matplotlib
            streamline_density : Densite des lignes de champ
            quiver_step : Pas pour les fleches d'orientation
        
        Returns:
            Axes matplotlib
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extraire champs
        U, V = self.get_field_components()
        magnitude = np.sqrt(U**2 + V**2)
        
        # 1. Fond : magnitude
        im = ax.imshow(magnitude.T, origin='lower',
                      extent=[0, self.nx, 0, self.ny],
                      cmap='hot', alpha=0.5, aspect='equal')
        plt.colorbar(im, ax=ax, label=f'|{self.field}| (a.u.)')
        
        # 2. Streamlines
        x = np.linspace(0, self.nx, self.nx)
        y = np.linspace(0, self.ny, self.ny)
        try:
            ax.streamplot(x, y, U.T, V.T,
                         density=streamline_density,
                         color='cyan', linewidth=1.0,
                         arrowsize=1.2, alpha=0.7,
                         arrowstyle='->', minlength=0.2)
        except:
            pass  # Skip si probleme
        
        # 3. Quelques fleches pour direction
        x_q = np.arange(0, self.nx, quiver_step)
        y_q = np.arange(0, self.ny, quiver_step)
        X_q, Y_q = np.meshgrid(x_q, y_q, indexing='ij')
        U_q = U[::quiver_step, ::quiver_step]
        V_q = V[::quiver_step, ::quiver_step]
        
        ax.quiver(X_q, Y_q, U_q, V_q,
                 color='white', scale=50, width=0.005,
                 headwidth=3.5, headlength=4, alpha=0.7,
                 scale_units='xy')
        
        ax.set_xlabel('X (cellules)')
        ax.set_ylabel('Y (cellules)')
        ax.set_title(f'Champ {self.field} - Vue hybride')
        ax.set_xlim(0, self.nx)
        ax.set_ylim(0, self.ny)
        
        return ax
    
    def plot_normalized(self,
                       ax: Optional[plt.Axes] = None,
                       step: int = 5,
                       arrow_scale: float = 3.0,
                       show_magnitude_bg: bool = True,
                       cmap: str = 'jet') -> plt.Axes:
        """
        Visualisation avec vecteurs normalises (meme longueur).
        Tous les vecteurs ont la meme taille mais sont colores par magnitude.
        Ideal pour voir les directions sans biais par l'amplitude.
        
        Args:
            ax : Axes matplotlib
            step : Pas d'echantillonnage des fleches
            arrow_scale : Facteur d'echelle des fleches normalisees (1.0-5.0, defaut=3.0)
            show_magnitude_bg : Afficher fond de magnitude
            cmap : Colormap pour les fleches
        
        Returns:
            Axes matplotlib
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extraire champs
        U, V = self.get_field_components()
        
        # Sous-echantillonnage
        U_sub = U[::step, ::step]
        V_sub = V[::step, ::step]
        
        # Grille
        x = np.arange(0, self.nx, step)
        y = np.arange(0, self.ny, step)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Magnitude AVANT normalisation
        magnitude = np.sqrt(U**2 + V**2)
        magnitude_sub = np.sqrt(U_sub**2 + V_sub**2)
        
        # Normaliser les vecteurs (longueur = 1)
        norm = magnitude_sub + 1e-12  # Eviter division par 0
        U_norm = U_sub / norm
        V_norm = V_sub / norm
        
        # Fond : magnitude complete
        if show_magnitude_bg:
            im = ax.imshow(magnitude.T, origin='lower',
                          extent=[0, self.nx, 0, self.ny],
                          cmap='gray', alpha=0.3, aspect='equal')
        
        # Fleches normalisees colorees par magnitude ORIGINALE
        # On utilise scale=1 et width adapte pour avoir des fleches uniformes
        quiv = ax.quiver(X, Y, U_norm, V_norm, magnitude_sub,
                        cmap=cmap, 
                        scale=1.0/arrow_scale,  # Controle taille uniforme
                        scale_units='xy',
                        angles='xy',
                        pivot='mid', 
                        alpha=0.9,
                        width=0.005, 
                        headwidth=3.5, 
                        headlength=4.5)
        
        plt.colorbar(quiv, ax=ax, label=f'|{self.field}| (a.u.)')
        
        ax.set_xlabel('X (cellules)')
        ax.set_ylabel('Y (cellules)')
        ax.set_title(f'Champ {self.field} - Vecteurs normalises')
        ax.set_xlim(0, self.nx)
        ax.set_ylim(0, self.ny)
        ax.set_aspect('equal')
        
        return ax


def compare_visualizations(sim, field='auto', z_index=0, save_path=None):
    """
    Creer une figure avec 3 modes de visualisation cote a cote.
    
    Args:
        sim : Instance Yee3D
        field : Type de champ ('E', 'H', ou 'auto' pour detection automatique)
        z_index : Indice z de la tranche
        save_path : Chemin pour sauvegarder (None = affichage)
    """
    viz = VectorFieldVisualizer(sim, field, z_index)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Mode 1 : Streamlines
    viz.plot_streamlines(axes[0], density=1.5, color_by_magnitude=True)
    
    # Mode 2 : Quiver
    viz.plot_quiver(axes[1], step=6, show_magnitude_bg=True)
    
    # Mode 3 : Hybride
    viz.plot_hybrid(axes[2], streamline_density=1.0, quiver_step=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def compare_all_modes(sim, field='auto', z_index=0, save_path=None):
    """
    Creer une figure avec TOUS les modes de visualisation (4 modes).
    
    Args:
        sim : Instance Yee3D
        field : Type de champ ('E', 'H', ou 'auto' pour detection automatique)
        z_index : Indice z de la tranche
        save_path : Chemin pour sauvegarder (None = affichage)
    """
    viz = VectorFieldVisualizer(sim, field, z_index)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    # Mode 1 : Streamlines
    viz.plot_streamlines(axes[0], density=0.8, color_by_magnitude=True)
    
    # Mode 2 : Quiver standard
    viz.plot_quiver(axes[1], step=6, show_magnitude_bg=True)
    
    # Mode 3 : Vecteurs normalises
    viz.plot_normalized(axes[2], step=5, arrow_scale=3.0, show_magnitude_bg=True)
    
    # Mode 4 : Hybride
    viz.plot_hybrid(axes[3], streamline_density=0.6, quiver_step=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


# Example usage
if __name__ == '__main__':
    print("Module de visualisation vectorielle des champs EM")
    print("Importer avec : from champs_v4.visualization.vector_field_viz import VectorFieldVisualizer")
