#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
Module de sondes électromagnétiques pour le simulateur FDTD Yee3D.

Ce module permet de placer des sondes virtuelles dans l'espace pour enregistrer
l'évolution temporelle des champs électromagnétiques et de l'énergie locale.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Physical constants
epsilon0 = 8.8541878128e-12
mu0 = 4 * np.pi * 1e-7


class FieldProbe:
    """
    Sonde pour enregistrer les champs EM à une position spécifique.
    
    Attributes:
        position (tuple): Position (ix, iy, iz) de la sonde dans la grille
        name (str): Nom de la sonde
        record_E (bool): Enregistrer les champs électriques
        record_H (bool): Enregistrer les champs magnétiques
        record_energy (bool): Enregistrer l'énergie locale
    """
    
    def __init__(self, position: Tuple[int, int, int], name: str = None,
                 record_E: bool = True, record_H: bool = True, 
                 record_energy: bool = True):
        """
        Initialise une sonde de champ.
        
        Parameters:
            position: (ix, iy, iz) indices de cellule dans la grille
            name: Nom optionnel pour identifier la sonde
            record_E: Enregistrer les composantes Ex, Ey, Ez
            record_H: Enregistrer les composantes Hx, Hy, Hz
            record_energy: Enregistrer l'énergie électrique et magnétique locale
        """
        self.position = position
        self.name = name or f"Probe_{position}"
        self.record_E = record_E
        self.record_H = record_H
        self.record_energy = record_energy
        
        # Historique temporel
        self.times: List[float] = []
        
        # Champs électriques
        self.Ex_history: List[float] = []
        self.Ey_history: List[float] = []
        self.Ez_history: List[float] = []
        
        # Champs magnétiques
        self.Hx_history: List[float] = []
        self.Hy_history: List[float] = []
        self.Hz_history: List[float] = []
        
        # Énergie locale
        self.U_electric_history: List[float] = []
        self.U_magnetic_history: List[float] = []
        self.U_total_history: List[float] = []
        
        logger.info(f"Created probe '{self.name}' at position {position}")
    
    def sample(self, sim, time: float):
        """
        Échantillonne les champs à la position de la sonde.
        
        Parameters:
            sim: Instance de Yee3D
            time: Temps actuel de la simulation
        """
        ix, iy, iz = self.position
        self.times.append(time)
        
        # Interpoler les champs E au centre de la cellule
        if self.record_E:
            # Ex est à (i, j+0.5, k+0.5)
            Ex = 0.25 * (
                sim.Ex[ix, iy, iz] + sim.Ex[ix, iy+1, iz] +
                sim.Ex[ix, iy, iz+1] + sim.Ex[ix, iy+1, iz+1]
            ) if iy < sim.ny and iz < sim.nz else 0.0
            
            # Ey est à (i+0.5, j, k+0.5)
            Ey = 0.25 * (
                sim.Ey[ix, iy, iz] + sim.Ey[ix+1, iy, iz] +
                sim.Ey[ix, iy, iz+1] + sim.Ey[ix+1, iy, iz+1]
            ) if ix < sim.nx and iz < sim.nz else 0.0
            
            # Ez est à (i+0.5, j+0.5, k)
            Ez = 0.25 * (
                sim.Ez[ix, iy, iz] + sim.Ez[ix+1, iy, iz] +
                sim.Ez[ix, iy+1, iz] + sim.Ez[ix+1, iy+1, iz]
            ) if ix < sim.nx and iy < sim.ny else 0.0
            
            self.Ex_history.append(Ex)
            self.Ey_history.append(Ey)
            self.Ez_history.append(Ez)
        
        # Interpoler les champs H au centre de la cellule
        if self.record_H:
            # Hx est à (i+0.5, j, k)
            Hx = 0.5 * (sim.Hx[ix, iy, iz] + sim.Hx[ix+1, iy, iz]) if ix < sim.nx else 0.0
            
            # Hy est à (i, j+0.5, k)
            Hy = 0.5 * (sim.Hy[ix, iy, iz] + sim.Hy[ix, iy+1, iz]) if iy < sim.ny else 0.0
            
            # Hz est à (i, j, k+0.5)
            Hz = 0.5 * (sim.Hz[ix, iy, iz] + sim.Hz[ix, iy, iz+1]) if iz < sim.nz else 0.0
            
            self.Hx_history.append(Hx)
            self.Hy_history.append(Hy)
            self.Hz_history.append(Hz)
        
        # Calculer l'énergie locale
        if self.record_energy and self.record_E and self.record_H:
            Ex = self.Ex_history[-1]
            Ey = self.Ey_history[-1]
            Ez = self.Ez_history[-1]
            Hx = self.Hx_history[-1]
            Hy = self.Hy_history[-1]
            Hz = self.Hz_history[-1]
            
            E2 = Ex**2 + Ey**2 + Ez**2
            H2 = Hx**2 + Hy**2 + Hz**2
            
            # Densité d'énergie volumique (J/m³)
            epsilon_r = sim.epsilon_r[ix, iy, iz] if ix < sim.nx and iy < sim.ny and iz < sim.nz else 1.0
            u_electric = 0.5 * epsilon0 * epsilon_r * E2
            u_magnetic = 0.5 * mu0 * H2
            
            self.U_electric_history.append(u_electric)
            self.U_magnetic_history.append(u_magnetic)
            self.U_total_history.append(u_electric + u_magnetic)
    
    def get_data(self) -> Dict[str, np.ndarray]:
        """
        Récupère toutes les données enregistrées sous forme de arrays NumPy.
        
        Returns:
            dict: Dictionnaire contenant les séries temporelles
        """
        data = {'time': np.array(self.times)}
        
        if self.record_E:
            data['Ex'] = np.array(self.Ex_history)
            data['Ey'] = np.array(self.Ey_history)
            data['Ez'] = np.array(self.Ez_history)
            data['E_magnitude'] = np.sqrt(data['Ex']**2 + data['Ey']**2 + data['Ez']**2)
        
        if self.record_H:
            data['Hx'] = np.array(self.Hx_history)
            data['Hy'] = np.array(self.Hy_history)
            data['Hz'] = np.array(self.Hz_history)
            data['H_magnitude'] = np.sqrt(data['Hx']**2 + data['Hy']**2 + data['Hz']**2)
        
        if self.record_energy:
            data['U_electric'] = np.array(self.U_electric_history)
            data['U_magnetic'] = np.array(self.U_magnetic_history)
            data['U_total'] = np.array(self.U_total_history)
        
        return data
    
    def compute_fft(self, field='Ez', window='hann'):
        """
        Calcule la transformée de Fourier du signal enregistré.
        
        Parameters:
            field (str): Champ à analyser ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 
                        'E_magnitude', 'H_magnitude')
            window (str): Type de fenêtrage ('hann', 'hamming', 'blackman', None)
        
        Returns:
            dict: Contient 'frequencies' (Hz), 'amplitude', 'phase', 'power_spectrum'
        
        Example:
            fft_data = probe.compute_fft('Ez')
            plt.plot(fft_data['frequencies'], fft_data['amplitude'])
        """
        # Récupérer les données
        data = self.get_data()
        
        if field not in data:
            raise ValueError(f"Field '{field}' not available. Recorded fields: {list(data.keys())}")
        
        signal = data[field]
        time = data['time']
        
        if len(signal) < 2:
            raise ValueError("Not enough samples for FFT (need at least 2)")
        
        # Calculer le pas de temps (suppose régulier)
        dt = np.mean(np.diff(time))
        fs = 1.0 / dt  # Fréquence d'échantillonnage
        
        # Appliquer une fenêtre pour réduire les fuites spectrales
        if window == 'hann':
            window_func = np.hanning(len(signal))
        elif window == 'hamming':
            window_func = np.hamming(len(signal))
        elif window == 'blackman':
            window_func = np.blackman(len(signal))
        else:
            window_func = np.ones(len(signal))
        
        signal_windowed = signal * window_func
        
        # FFT
        fft_result = np.fft.rfft(signal_windowed)
        frequencies = np.fft.rfftfreq(len(signal), dt)
        
        # Amplitude (normalisée)
        amplitude = np.abs(fft_result) * 2.0 / len(signal)
        
        # Phase
        phase = np.angle(fft_result)
        
        # Densité spectrale de puissance
        power_spectrum = (np.abs(fft_result) ** 2) / len(signal)
        
        return {
            'frequencies': frequencies,
            'amplitude': amplitude,
            'phase': phase,
            'power_spectrum': power_spectrum,
            'fft_complex': fft_result,
            'window': window,
            'sampling_rate': fs,
            'field': field
        }
    
    def get_dominant_frequencies(self, field='Ez', n_peaks=5, window='hann'):
        """
        Trouve les fréquences dominantes dans le signal.
        
        Parameters:
            field (str): Champ à analyser
            n_peaks (int): Nombre de pics à retourner
            window (str): Type de fenêtrage
        
        Returns:
            list: Liste de tuples (fréquence_Hz, amplitude)
        """
        fft_data = self.compute_fft(field, window)
        
        frequencies = fft_data['frequencies']
        amplitude = fft_data['amplitude']
        
        # Trouver les pics (ignorer DC: freq=0)
        # Chercher les maxima locaux
        from scipy import signal as sp_signal
        try:
            peaks, properties = sp_signal.find_peaks(amplitude[1:], height=0)
            peaks = peaks + 1  # Ajuster l'index (on a ignoré le premier)
            
            # Trier par amplitude décroissante
            peak_amplitudes = amplitude[peaks]
            sorted_indices = np.argsort(peak_amplitudes)[::-1]
            
            # Retourner les n_peaks plus grands
            top_peaks = sorted_indices[:n_peaks]
            result = [(frequencies[peaks[i]], amplitude[peaks[i]]) for i in top_peaks]
            
        except ImportError:
            # Si scipy n'est pas disponible, méthode simple
            # Ignorer la composante DC (index 0)
            sorted_indices = np.argsort(amplitude[1:])[::-1]
            sorted_indices = sorted_indices + 1  # Ajuster pour DC ignoré
            result = [(frequencies[i], amplitude[i]) for i in sorted_indices[:n_peaks]]
        
        return result
    
    def compute_spectrogram(self, field='Ez', nperseg=256, noverlap=None):
        """
        Calcule le spectrogramme (FFT glissante) pour observer l'évolution fréquentielle.
        
        Parameters:
            field (str): Champ à analyser
            nperseg (int): Longueur de chaque segment pour la FFT
            noverlap (int): Nombre de points de recouvrement entre segments
        
        Returns:
            dict: Contient 'times', 'frequencies', 'spectrogram' (matrice 2D)
        """
        data = self.get_data()
        
        if field not in data:
            raise ValueError(f"Field '{field}' not available")
        
        signal = data[field]
        time = data['time']
        dt = np.mean(np.diff(time))
        fs = 1.0 / dt
        
        if noverlap is None:
            noverlap = nperseg // 2
        
        # Utiliser scipy si disponible
        try:
            from scipy import signal as sp_signal
            f, t, Sxx = sp_signal.spectrogram(signal, fs, nperseg=nperseg, noverlap=noverlap)
            
            return {
                'times': t,
                'frequencies': f,
                'spectrogram': Sxx,
                'field': field
            }
        except ImportError:
            logger.warning("scipy not available, spectrogram computation skipped")
            return None
    
    def clear(self):
        """Efface l'historique de la sonde."""
        self.times.clear()
        self.Ex_history.clear()
        self.Ey_history.clear()
        self.Ez_history.clear()
        self.Hx_history.clear()
        self.Hy_history.clear()
        self.Hz_history.clear()
        self.U_electric_history.clear()
        self.U_magnetic_history.clear()
        self.U_total_history.clear()
    
    def __repr__(self):
        return f"FieldProbe('{self.name}', position={self.position}, samples={len(self.times)})"


class ProbeManager:
    """
    Gestionnaire de sondes pour faciliter l'utilisation de multiples sondes.
    """
    
    def __init__(self):
        """Initialise le gestionnaire de sondes."""
        self.probes: Dict[str, FieldProbe] = {}
        logger.info("ProbeManager initialized")
    
    def add_probe(self, position: Tuple[int, int, int], name: str = None,
                  record_E: bool = True, record_H: bool = True,
                  record_energy: bool = True) -> FieldProbe:
        """
        Ajoute une nouvelle sonde.
        
        Parameters:
            position: (ix, iy, iz) position dans la grille
            name: Nom de la sonde (auto-généré si None)
            record_E: Enregistrer les champs E
            record_H: Enregistrer les champs H
            record_energy: Enregistrer l'énergie locale
        
        Returns:
            FieldProbe: La sonde créée
        """
        if name is None:
            name = f"Probe_{len(self.probes)}"
        
        if name in self.probes:
            logger.warning(f"Probe '{name}' already exists, replacing it")
        
        probe = FieldProbe(position, name, record_E, record_H, record_energy)
        self.probes[name] = probe
        return probe
    
    def add_line_of_probes(self, start: Tuple[int, int, int], 
                          end: Tuple[int, int, int], 
                          num_probes: int,
                          name_prefix: str = "line",
                          **kwargs) -> List[FieldProbe]:
        """
        Ajoute une ligne de sondes entre deux points.
        
        Parameters:
            start: Point de départ (ix, iy, iz)
            end: Point d'arrivée (ix, iy, iz)
            num_probes: Nombre de sondes sur la ligne
            name_prefix: Préfixe pour les noms des sondes
            **kwargs: Arguments additionnels pour FieldProbe
        
        Returns:
            List[FieldProbe]: Liste des sondes créées
        """
        probes = []
        for i in range(num_probes):
            t = i / max(1, num_probes - 1) if num_probes > 1 else 0.0
            pos = tuple(int(s + t * (e - s)) for s, e in zip(start, end))
            name = f"{name_prefix}_{i}"
            probe = self.add_probe(pos, name, **kwargs)
            probes.append(probe)
        
        logger.info(f"Added {num_probes} probes from {start} to {end}")
        return probes
    
    def add_plane_of_probes(self, axis: str, position: int,
                           x_range: Tuple[int, int],
                           y_range: Tuple[int, int],
                           spacing: int = 1,
                           name_prefix: str = "plane",
                           **kwargs) -> List[FieldProbe]:
        """
        Ajoute un plan de sondes perpendiculaire à un axe.
        
        Parameters:
            axis: Axe normal au plan ('x', 'y', ou 'z')
            position: Position le long de l'axe
            x_range: (min, max) pour la première dimension du plan
            y_range: (min, max) pour la deuxième dimension du plan
            spacing: Espacement entre les sondes
            name_prefix: Préfixe pour les noms
            **kwargs: Arguments additionnels pour FieldProbe
        
        Returns:
            List[FieldProbe]: Liste des sondes créées
        """
        probes = []
        count = 0
        
        for i in range(x_range[0], x_range[1], spacing):
            for j in range(y_range[0], y_range[1], spacing):
                if axis == 'x':
                    pos = (position, i, j)
                elif axis == 'y':
                    pos = (i, position, j)
                elif axis == 'z':
                    pos = (i, j, position)
                else:
                    raise ValueError(f"Invalid axis: {axis}")
                
                name = f"{name_prefix}_{count}"
                probe = self.add_probe(pos, name, **kwargs)
                probes.append(probe)
                count += 1
        
        logger.info(f"Added {count} probes in {axis}-plane at position {position}")
        return probes
    
    def sample_all(self, sim, time: float):
        """
        Échantillonne toutes les sondes.
        
        Parameters:
            sim: Instance Yee3D
            time: Temps actuel
        """
        for probe in self.probes.values():
            probe.sample(sim, time)
    
    def get_probe(self, name: str) -> Optional[FieldProbe]:
        """Récupère une sonde par son nom."""
        return self.probes.get(name)
    
    def get_all_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Récupère les données de toutes les sondes.
        
        Returns:
            dict: {nom_sonde: données}
        """
        return {name: probe.get_data() for name, probe in self.probes.items()}
    
    def clear_all(self):
        """Efface l'historique de toutes les sondes."""
        for probe in self.probes.values():
            probe.clear()
        logger.info("Cleared all probe histories")
    
    def remove_probe(self, name: str):
        """Supprime une sonde."""
        if name in self.probes:
            del self.probes[name]
            logger.info(f"Removed probe '{name}'")
    
    def list_probes(self) -> List[str]:
        """Liste les noms de toutes les sondes."""
        return list(self.probes.keys())
    
    def __len__(self):
        return len(self.probes)
    
    def __repr__(self):
        return f"ProbeManager({len(self.probes)} probes)"


def save_probe_data(probe: FieldProbe, filename: str):
    """
    Sauvegarde les données d'une sonde dans un fichier NPZ.
    
    Parameters:
        probe: Sonde à sauvegarder
        filename: Nom du fichier de sortie (.npz)
    """
    data = probe.get_data()
    data['probe_name'] = probe.name
    data['probe_position'] = np.array(probe.position)
    np.savez(filename, **data)
    logger.info(f"Saved probe '{probe.name}' data to {filename}")


def load_probe_data(filename: str) -> Dict[str, np.ndarray]:
    """
    Charge les données d'une sonde depuis un fichier NPZ.
    
    Parameters:
        filename: Fichier à charger
    
    Returns:
        dict: Données de la sonde
    """
    data = np.load(filename, allow_pickle=True)
    logger.info(f"Loaded probe data from {filename}")
    return dict(data)


if __name__ == '__main__':
    print("Field probes module for Yee3D FDTD simulator")
    print("Usage:")
    print("  from champs_v4.probes import ProbeManager")
    print("  manager = ProbeManager()")
    print("  manager.add_probe((nx//2, ny//2, nz//2), 'center')")
    print("  # In simulation loop: manager.sample_all(sim, t)")
