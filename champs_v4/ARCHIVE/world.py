import pygame
import numpy as np
import constants as const
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import tqdm
import sys
import shutil
import datetime
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
import logging
from utils import print_debug, debug 
import random as rd
from corps import Particle
from concurrent.futures import ThreadPoolExecutor
import time
from scipy.fft import fftn, ifftn
import threading
import discord
try:
    import numba
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False
    njit = None
    prange = None
import math


if _NUMBA_AVAILABLE:
    # define a numba-accelerated kernel to compute the electric field on a
    # flattened set of points. This function expects plain numpy arrays.
    @njit(parallel=True)
    def _calc_E_numba(flat_x, flat_y, flat_z, part_x, part_y, part_z, part_q, k):
        npoints = flat_x.shape[0]
        npart = part_x.shape[0]
        out = np.zeros((npoints, 3), dtype=np.float64)
        for i in prange(npoints):
            x = flat_x[i]
            y = flat_y[i]
            z = flat_z[i]
            rx = 0.0
            ry = 0.0
            rz = 0.0
            for j in range(npart):
                dx = x - part_x[j]
                dy = y - part_y[j]
                dz = z - part_z[j]
                r2 = dx * dx + dy * dy + dz * dz + 1e-12
                invr3 = 1.0 / (r2 * math.sqrt(r2))
                coef = k * part_q[j] * invr3
                rx += coef * dx
                ry += coef * dy
                rz += coef * dz
            out[i, 0] = rx
            out[i, 1] = ry
            out[i, 2] = rz
        return out
else:
    _calc_E_numba = None

"""
World simulation module

This module defines the `World` class which simulates electric and
magnetic fields on a 3D grid and a collection of charged particles.
It supports both an FDTD-like finite-difference update and spectral
calculations via FFT. The code has been adjusted to reduce reliance on
module-level globals and to be safer for small-scale use.
"""


class World:
    def step_FDTD(self):
        """
        Effectue un pas de temps FDTD complet :
        1. Calcule la densité de courant J
        2. Met à jour B (Faraday)
        3. Met à jour E (Ampère-Maxwell)
        4. Met à jour les particules
        """
        J = self.compute_current_density_J()
        self.update_B_FDTD()
        self.update_E_FDTD(J)
        for part in self.particles:
            part.calc_next(self.field_E, self.field_B, self.size, self.dt, self.cell_size, fil=self.field_E_fil)
        self.temps += self.dt
    def compute_current_density_J(self):
        """
        Calcule la densité de courant J (vecteur) sur la grille à partir des particules.
        Retourne un tableau numpy de même forme que self.field_E (x, y, z, 3).
        """
        # Start from any macroscopic source currently defined on the grid
        J = np.array(self.J_source, copy=True)
        # Add particle-based contributions (backwards compatible)
        for part in self.particles:
            ix = int(part.x / self.cell_size)
            iy = int(part.y / self.cell_size)
            iz = int(part.z / self.cell_size)
            # Vérification des bornes
            if 0 <= ix < J.shape[0] and 0 <= iy < J.shape[1] and 0 <= iz < J.shape[2]:
                # J contribution from a particle: charge * velocity (coarse deposition)
                J[ix, iy, iz, 0] += part.charge * part.vx
                J[ix, iy, iz, 1] += part.charge * part.vy
                J[ix, iy, iz, 2] += part.charge * part.vz
        return J
    def update_B_FDTD(self):
        """
        Met à jour le champ magnétique B selon la loi de Faraday (FDTD):
        dB/dt = -curl(E)
        """
        Ex = self.field_E[..., 0]
        Ey = self.field_E[..., 1]
        Ez = self.field_E[..., 2]
        dx = self.cell_size
        dt = self.dt
        # Calcul du rotationnel de E (différences finies centrées)
        curlE_x = (np.roll(Ez, -1, axis=1) - np.roll(Ez, 1, axis=1))/(2*dx) - (np.roll(Ey, -1, axis=2) - np.roll(Ey, 1, axis=2))/(2*dx)
        curlE_y = (np.roll(Ex, -1, axis=2) - np.roll(Ex, 1, axis=2))/(2*dx) - (np.roll(Ez, -1, axis=0) - np.roll(Ez, 1, axis=0))/(2*dx)
        curlE_z = (np.roll(Ey, -1, axis=0) - np.roll(Ey, 1, axis=0))/(2*dx) - (np.roll(Ex, -1, axis=1) - np.roll(Ex, 1, axis=1))/(2*dx)
        self.field_B[..., 0] += -dt * curlE_x
        self.field_B[..., 1] += -dt * curlE_y
        self.field_B[..., 2] += -dt * curlE_z

    def update_E_FDTD(self, J=None):
        """
        Met à jour le champ électrique E selon la loi d'Ampère-Maxwell (FDTD):
        dE/dt = (1/epsilon_0) * (curl(B) - J)
        """
        Bx = self.field_B[..., 0]
        By = self.field_B[..., 1]
        Bz = self.field_B[..., 2]
        dx = self.cell_size
        dt = self.dt
        epsilon0 = const.epsilon_0
        # Calcul du rotationnel de B (différences finies centrées)
        curlB_x = (np.roll(Bz, -1, axis=1) - np.roll(Bz, 1, axis=1))/(2*dx) - (np.roll(By, -1, axis=2) - np.roll(By, 1, axis=2))/(2*dx)
        curlB_y = (np.roll(Bx, -1, axis=2) - np.roll(Bx, 1, axis=2))/(2*dx) - (np.roll(Bz, -1, axis=0) - np.roll(Bz, 1, axis=0))/(2*dx)
        curlB_z = (np.roll(By, -1, axis=0) - np.roll(By, 1, axis=0))/(2*dx) - (np.roll(Bx, -1, axis=1) - np.roll(Bx, 1, axis=1))/(2*dx)
        if J is None:
            J = np.zeros_like(self.field_E)
        self.field_E[..., 0] += dt/epsilon0 * (curlB_x - J[..., 0])
        self.field_E[..., 1] += dt/epsilon0 * (curlB_y - J[..., 1])
        self.field_E[..., 2] += dt/epsilon0 * (curlB_z - J[..., 2])
    def __init__(
        self,
        size,
        cell_size,
        dt: float,
        I=0.0,
        U=0.0,
        type_de_courant="cc",
        f=0.0,
        axe="x",
        position_x=0,
        position_y=0,
        type_simulation="",
        centre_x=0.0,
        centre_y=0.0,
        centre_z=0.0,
        longueur=0.0,
        duree_simulation=0.0,
    ) -> None:
        """
        Initialize simulation world and grid fields.

        Many configuration options can be provided; defaults are present so
        the constructor can be called with only the required geometric args.
        """
        self.size: float = float(size)
        self.dt: float = float(dt)
        self.cell_size = float(cell_size)

        size_int = int(self.size / self.cell_size)
        self.field_E = np.zeros((size_int, size_int, size_int, 3), dtype=np.float64)
        self.field_E_fil = np.zeros((size_int, size_int, size_int, 3), dtype=np.float64)

        self.field_B = np.zeros((size_int, size_int, size_int, 3), dtype=np.float64)

        # macroscopic current source (A/m^2 or consistent units used in update)
        self.J_source = np.zeros((size_int, size_int, size_int, 3), dtype=np.float64)

        self.particles: list[Particle] = []

        self.temps = 0.0

        # electrical parameters
        self.U = float(U)
        self.I = float(I)

        # simulation configuration (prefer instance attributes over globals)
        self.type_de_courant = type_de_courant
        self.f = float(f)
        self.axe = axe
        self.position_x = int(position_x)
        self.position_y = int(position_y)
        self.type_simulation = type_simulation

        # geometry helpers
        self.centre_x = float(centre_x)
        self.centre_y = float(centre_y)
        self.centre_z = float(centre_z)
        self.longueur = float(longueur)

        # duration used by add_fil when computing particle charges
        self.duree_simulation = float(duree_simulation)

        # Attempt to use Numba-accelerated routines when available
        self._use_numba = _NUMBA_AVAILABLE

        # CFL stability check for explicit FDTD-like update; adjust dt if
        # it's too large for electromagnetic waves (approximate condition)
        try:
            c = const.c
            cfl = self.cell_size / (c * math.sqrt(3.0))
            if self.dt > cfl:
                print_debug(
                    f"Warning: dt ({self.dt}) exceeds CFL estimate ({cfl:.3e}). Reducing dt to 0.9*CFL."
                )
                self.dt = 0.9 * cfl
        except Exception:
            # If constants missing or other issue, silently skip
            pass
        
        
        
        

    def add_part(self, part: Particle) -> None:
        self.particles.append(part)

    def add_fil(self, axe, position_x, position_y, d : int):
        """
        Backwards-compatible filament creator. By default this creates a
        macroscopic current density on the grid (call to `add_macro_fil`). If
        `use_particles=True` is passed, it will also create discrete particles
        along the filament (legacy behaviour).
        """
        use_particles = False
        if isinstance(d, dict):
            # allow passing options as dict: {'d': value, 'use_particles': True}
            opts = d
            d = opts.get('d', 1)
            use_particles = opts.get('use_particles', False)

        # create macroscopic current distribution
        self.add_macro_fil(axe, position_x, position_y, d=d)

        # legacy: optionally create discrete particles (kept for compatibility)
        if use_particles:
            # replicate minimal previous behaviour: create particles carrying a
            # fraction of the macroscopic current over the simulation duration.
            norm_E = (self.I**2) / (self.U * const.epsilon_0) if self.U != 0 else 0.0
            n_p = d * int(self.size / self.cell_size)
            duree = getattr(self, 'duree_simulation', 0.0)
            q = (self.I * duree) / n_p if n_p > 0 else 0.0
            masse = (q / const.charge_electron) * const.masse_electron
            if axe == 'x':
                for i in range(n_p):
                    x = (i * self.cell_size) / d
                    y = position_x * self.cell_size
                    z = position_y * self.cell_size
                    self.add_part(Particle(x, y, z, const.charge_electron, masse, fil=True))
            elif axe == 'y':
                for i in range(n_p):
                    x = position_x * self.cell_size
                    y = (i * self.cell_size) / d
                    z = position_y * self.cell_size
                    self.add_part(Particle(x, y, z, const.charge_electron, masse, fil=True))
            elif axe == 'z':
                for i in range(n_p):
                    x = position_x * self.cell_size
                    y = position_y * self.cell_size
                    z = i * self.cell_size
                    self.add_part(Particle(x, y, z, const.charge_electron, masse, fil=True))


    
    
    def calc_E2(self) -> None:

        shape = self.field_E.shape
        x_coords, y_coords, z_coords = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
        )

        x_coords = x_coords * self.cell_size
        y_coords = y_coords * self.cell_size
        z_coords = z_coords * self.cell_size

        # For memory-safety and simplicity, reuse the single-threaded
        # implementation in `calc_E`. The threaded variant allocated a
        # full-sized array per particle which is memory-heavy.
        self.calc_E()

    def calc_E(self):
        """
        Compute the electric field by summing Coulomb contributions from each
        particle. This implementation processes the grid in blocks to keep
        memory usage bounded for larger grids.

        Accuracy: the field is computed as E = k*q * r_vec / r^3 with a small
        epsilon to avoid division by zero at r=0.
        """
        shape = self.field_E.shape
        nx, ny, nz = shape[0], shape[1], shape[2]

        # Precompute cell centers coordinates
        x_coords = (np.arange(nx) * self.cell_size).astype(np.float64)
        y_coords = (np.arange(ny) * self.cell_size).astype(np.float64)
        z_coords = (np.arange(nz) * self.cell_size).astype(np.float64)

        total_E = np.zeros_like(self.field_E)

        # Choose a block size for the flattened grid to control memory use.
        # Tune this depending on available memory. A block of 4096 is small;
        # for large grids you may reduce it.
        block_size = 4096

        # Flatten grid indices and loop in blocks
        grid_x, grid_y, grid_z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
        flat_x = grid_x.ravel()
        flat_y = grid_y.ravel()
        flat_z = grid_z.ravel()
        npoints = flat_x.size

        eps = 1e-12
        # If Numba is available and enabled use it for the heavy inner loop.
        if self._use_numba:
            try:
                # prepare particle arrays
                part_x = np.array([p.x for p in self.particles], dtype=np.float64)
                part_y = np.array([p.y for p in self.particles], dtype=np.float64)
                part_z = np.array([p.z for p in self.particles], dtype=np.float64)
                part_q = np.array([p.charge for p in self.particles], dtype=np.float64)
                # call numba-accelerated kernel
                flat_out = _calc_E_numba(flat_x, flat_y, flat_z, part_x, part_y, part_z, part_q, const.k)
                # scatter into total_E
                ix = (np.arange(npoints) // (ny * nz))
                rem = np.arange(npoints) % (ny * nz)
                iy = rem // nz
                iz = rem % nz
                total_E[ix, iy, iz, 0] = flat_out[:, 0]
                total_E[ix, iy, iz, 1] = flat_out[:, 1]
                total_E[ix, iy, iz, 2] = flat_out[:, 2]
                # zero fields at particle grid cells
                for part in self.particles:
                    px = int(part.x / self.cell_size)
                    py = int(part.y / self.cell_size)
                    pz = int(part.z / self.cell_size)
                    if 0 <= px < nx and 0 <= py < ny and 0 <= pz < nz:
                        total_E[px, py, pz, :] = 0.0
                self.field_E = total_E
                if np.isnan(self.field_E).any():
                    print_debug("Le champ électrique contient des valeurs NaN après le calcul.")
                return
            except Exception:
                # fallback to block loop if numba call fails
                pass

        for start in range(0, npoints, block_size):
            end = min(start + block_size, npoints)
            bx = flat_x[start:end]
            by = flat_y[start:end]
            bz = flat_z[start:end]

            # accumulator for this block
            block_E = np.zeros((end - start, 3), dtype=np.float64)

            for part in self.particles:
                rx = bx - part.x
                ry = by - part.y
                rz = bz - part.z
                r2 = rx * rx + ry * ry + rz * rz + eps
                inv_r3 = 1.0 / (r2 * np.sqrt(r2))
                coef = const.k * part.charge * inv_r3
                block_E[:, 0] += coef * rx
                block_E[:, 1] += coef * ry
                block_E[:, 2] += coef * rz

            # write block back into total_E (reshape indices)
            idxs = np.arange(start, end)
            ix = (idxs // (ny * nz))
            rem = idxs % (ny * nz)
            iy = rem // nz
            iz = rem % nz

            total_E[ix, iy, iz, 0] = block_E[:, 0]
            total_E[ix, iy, iz, 1] = block_E[:, 1]
            total_E[ix, iy, iz, 2] = block_E[:, 2]

        # Ensure the field at exact particle grid positions is zeroed
        for part in self.particles:
            px = int(part.x / self.cell_size)
            py = int(part.y / self.cell_size)
            pz = int(part.z / self.cell_size)
            if 0 <= px < nx and 0 <= py < ny and 0 <= pz < nz:
                total_E[px, py, pz, :] = 0.0

        self.field_E = total_E
        if np.isnan(self.field_E).any():
            print_debug("Le champ électrique contient des valeurs NaN après le calcul.")

    def _calculate_properties_in_cell(self, particle: "Particle") -> tuple:
        """
        Calculate velocity vector and charge for a given particle.

        Args:
            particle (Particle): Particle for which to calculate properties.

        Returns:
            tuple: Tuple containing the velocity vector (vx, vy, vz) and charge of the particle.
        """
        velocity = (particle.vx, particle.vy, particle.vz)
        return velocity, particle.charge

    def calculate_properties_in_cells(self) -> tuple:
        """
        Calculate the average velocity vector and charge of particles in each cell.

        Returns:
            tuple: Tuple containing arrays of average velocity vector and charge in each cell.
        """
        # Initialize arrays to store the sum of velocity vector and charge for each cell
        cell_velocity_sum = np.zeros(
            (self.field_E.shape[0], self.field_E.shape[1], self.field_E.shape[2], 3),
            dtype=np.float64,
        )
        cell_charge_sum = np.zeros_like(self.field_E[..., 0], dtype=np.float64)
        cell_particle_count = np.zeros_like(self.field_E[..., 0], dtype=int)

        # Vectorized accumulation to avoid race conditions with threads.
        nx, ny, nz = self.field_E.shape[:3]

        if len(self.particles) == 0:
            return cell_velocity_sum, cell_charge_sum

        # Gather particle properties into arrays
        xs = np.array([p.x for p in self.particles])
        ys = np.array([p.y for p in self.particles])
        zs = np.array([p.z for p in self.particles])
        vxs = np.array([p.vx for p in self.particles])
        vys = np.array([p.vy for p in self.particles])
        vzs = np.array([p.vz for p in self.particles])
        charges = np.array([p.charge for p in self.particles])

        # Convert positions to cell indices (clamped to domain)
        ixs = np.clip((xs / self.cell_size).astype(int), 0, nx - 1)
        iys = np.clip((ys / self.cell_size).astype(int), 0, ny - 1)
        izs = np.clip((zs / self.cell_size).astype(int), 0, nz - 1)

        # Flatten index for bincount/add.at accumulation
        flat_idx = (ixs * ny + iys) * nz + izs
        n_cells = nx * ny * nz

        # Flat accumulators
        flat_vel_sum = np.zeros((n_cells, 3), dtype=np.float64)
        flat_charge_sum = np.zeros(n_cells, dtype=np.float64)

        # Accumulate velocity components and charges into flattened arrays
        np.add.at(flat_vel_sum[:, 0], flat_idx, vxs)
        np.add.at(flat_vel_sum[:, 1], flat_idx, vys)
        np.add.at(flat_vel_sum[:, 2], flat_idx, vzs)
        np.add.at(flat_charge_sum, flat_idx, charges)

        # Reshape back to 3D grid
        cell_velocity_sum = flat_vel_sum.reshape((nx, ny, nz, 3))
        cell_charge_sum = flat_charge_sum.reshape((nx, ny, nz))

        # Build particle counts for averaging
        flat_counts = np.bincount(flat_idx, minlength=n_cells).reshape((nx, ny, nz))

        # Compute averages safely (avoid division by zero)
        cell_average_velocity = np.zeros_like(cell_velocity_sum)
        cell_average_charge = np.zeros_like(cell_charge_sum)
        mask = flat_counts > 0
        if np.any(mask):
            cell_average_velocity[mask] = (
                cell_velocity_sum[mask] / flat_counts[mask][..., np.newaxis]
            )
            cell_average_charge[mask] = cell_charge_sum[mask] / flat_counts[mask]

        return cell_average_velocity, cell_average_charge

    def calculate_current_density(self):
        """
        Calculate the current density in each cell based on particle velocities and charges.

        Returns:
            numpy.ndarray: Array containing the current density in each cell.
        """
        # Get average velocity vector and charge in each cell
        V, Q = self.calculate_properties_in_cells()

        # Current density is vector field: J = charge_density * velocity
        J = V * Q[..., np.newaxis]
        return J

    def calculate_rotationnel_E_fft(self):
        shape = self.field_E.shape
        rotationnel_E = np.zeros_like(self.field_E)

        # Transformée de Fourier du champ électrique
        E_fft = fftn(self.field_E, axes=(0, 1, 2))

        # Définition du vecteur d'onde
        kx = np.fft.fftfreq(shape[0], d=self.cell_size)
        ky = np.fft.fftfreq(shape[1], d=self.cell_size)
        kz = np.fft.fftfreq(shape[2], d=self.cell_size)

        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")

        # Calcul du spectre du rotationnel en espace des vecteurs d'onde
        rotationnel_E_fft_x = 1j * (KY * E_fft[..., 2] - KZ * E_fft[..., 1])
        rotationnel_E_fft_y = 1j * (KZ * E_fft[..., 0] - KX * E_fft[..., 2])
        rotationnel_E_fft_z = 1j * (KX * E_fft[..., 1] - KY * E_fft[..., 0])

        # Concaténation pour obtenir le spectre du rotationnel
        rotationnel_E_fft = np.stack(
            (rotationnel_E_fft_x, rotationnel_E_fft_y, rotationnel_E_fft_z), axis=-1
        )

        # Transformée inverse de Fourier pour obtenir le rotationnel dans l'espace physique
        rotationnel_E = ifftn(rotationnel_E_fft, axes=(0, 1, 2)).real

        return rotationnel_E
    
    def ca(self):
        """
        Apply an AC (sinusoidal) drive to the filament/source fields.
        Uses amplitude `self.U` and `self.I` (set at construction) and frequency
        `self.f` or global `f` if not present on the instance.
        """
        # Frequency: prefer instance attribute, else fallback to module global
        freq = getattr(self, "f", globals().get("f", 0.0))

        # Preserve amplitudes and compute instantaneous values
        U_amp = getattr(self, "U", 0.0)
        I_amp = getattr(self, "I", 0.0)
        # instantaneous values (overwrite instance for visualization)
        self.U = U_amp * np.sin(2 * np.pi * freq * self.temps)
        self.I = I_amp * np.sin(2 * np.pi * freq * self.temps)

        # Avoid division by zero when computing norm_E
        norm_E = (self.I ** 2) / (self.U * const.epsilon_0) if self.U != 0 else 0.0
        print_debug(f"norm_E={norm_E}, self.U={self.U}")

        # Determine axis and positions (prefer instance attributes)
        axe_local = getattr(self, "axe", globals().get("axe", "x"))
        posx = getattr(self, "position_x", globals().get("position_x", 0))
        posy = getattr(self, "position_y", globals().get("position_y", 0))

        if axe_local == "x":
            self.field_E_fil[:, posx, posy, 0] = norm_E
        elif axe_local == "y":
            self.field_E_fil[:, posx, posy, 0] = norm_E
        elif axe_local == "z":
            self.field_E_fil[:, posx, posy, 0] = norm_E

    def add_macro_fil(self, axe, position_x, position_y, d: int = 1, current: float = None, width_cells: int = 1):
        """
        Add a macroscopic filament current along axis `axe` at grid cell
        coordinates `(position_x, position_y)`. The current is distributed
        along the cells of the filament. `d` controls the number of segments
        (density) and `width_cells` spreads the current over neighboring cells.

        `current` defaults to `self.I` when not provided.
        """
        I_val = float(current) if current is not None else float(getattr(self, 'I', 0.0))
        nx, ny, nz = self.field_E.shape[:3]

        # define per-cell current density (A/m^2) approximated as current / area
        cell_area = self.cell_size ** 2
        J_per_cell = I_val / cell_area if cell_area > 0 else 0.0

        if axe == 'x':
            # filament extends along x-axis; position_x,pos_y are y,z indices
            for ix in range(nx):
                for wx in range(-(width_cells//2), width_cells//2 + 1):
                    iy = position_x + wx
                    for wy in range(-(width_cells//2), width_cells//2 + 1):
                        iz = position_y + wy
                        if 0 <= iy < ny and 0 <= iz < nz:
                            self.J_source[ix, iy, iz, 0] = J_per_cell
        elif axe == 'y':
            for iy in range(ny):
                for wx in range(-(width_cells//2), width_cells//2 + 1):
                    ix = position_x + wx
                    for wy in range(-(width_cells//2), width_cells//2 + 1):
                        iz = position_y + wy
                        if 0 <= ix < nx and 0 <= iz < nz:
                            self.J_source[ix, iy, iz, 1] = J_per_cell
        elif axe == 'z':
            for iz in range(nz):
                for wx in range(-(width_cells//2), width_cells//2 + 1):
                    ix = position_x + wx
                    for wy in range(-(width_cells//2), width_cells//2 + 1):
                        iy = position_y + wy
                        if 0 <= ix < nx and 0 <= iy < ny:
                            self.J_source[ix, iy, iz, 2] = J_per_cell
            
    
    
    def calc_B(self):
        # Compute curl(E) spectrally and advance B using a simple explicit
        # Euler step: dB/dt = -curl(E). A proper RK4 would require RHS
        # evaluations at intermediate states; keep explicit update for clarity.
        rotationnel_E = self.calculate_rotationnel_E_fft()
        self.field_B += -self.dt * rotationnel_E

    def calc_next(self):
        # Use instance value for current type if available, else fallback to global
        type_courant = getattr(self, "type_de_courant", globals().get("type_de_courant", None))
        if type_courant == "ca":
            self.ca()

        for part in self.particles:

            # Vérifier si les coordonnées de la particule restent dans les limites du monde simulé
            if (
                0 <= part.x < self.size
                and 0 <= part.y < self.size
                and 0 <= part.z < self.size
            ):
                # Si les coordonnées sont valides, calculer la prochaine position de la particule
                if not part.solen:
                    
                    part.calc_next(
                        self.field_E,
                        self.field_B,
                        self.size,
                        self.dt,
                        self.cell_size,
                        fil=self.field_E_fil,
                    )
                else:
                    liste_position = self.particles.index(part)+1
                    if  liste_position < len(self.particles):
                        next_part = self.particles[liste_position]
                    else:
                        next_part = self.particles[0]
                    
                    X = next_part.x-part.x
                    Y = next_part.y-part.y
                    Z = next_part.z-part.z
                    norm = np.sqrt((X)**2 +(Y)**2 +(Z)**2            )
                    direction_x = X/norm
                    direction_y = Y/norm
                    direction_z = Z/norm
                    # Use instance timestep
                    dt_local = self.dt
                    dx = part.vx * dt_local
                    dy = part.vy * dt_local
                    dz = part.vz * dt_local
                    part.x += dx*direction_x
                    part.y += dy*direction_y
                    if part.z > next_part.z:
                        part.z += dz
                    else:
                    
                        part.z += dz*direction_z
                        
                    
                        
                    #print(part.x,part.y,part.z)
                    
                    # Wrap/respawn positions: prefer instance attributes when present
                    centre_x_local = getattr(self, "centre_x", globals().get("centre_x", None))
                    centre_y_local = getattr(self, "centre_y", globals().get("centre_y", None))
                    centre_z_local = getattr(self, "centre_z", globals().get("centre_z", None))
                    longueur_local = getattr(self, "longueur", globals().get("longueur", 0))
                    if part.x >= self.size and centre_x_local is not None:
                        part.x = centre_x_local
                    if part.y >= self.size and centre_y_local is not None:
                        part.y = centre_y_local
                    if part.z >= (centre_z_local + longueur_local - 1 * self.cell_size) and centre_z_local is not None:
                        part.z = centre_z_local
                   
            else:
                # Si les coordonnées sont invalides, ignorer la mise à jour de la particule
                print_debug(
                    f"Attention : Les coordonnées de la particule sont hors des limites du monde simulé,  x = {part.x} / y = {part.y} / z = {part.z}."
                )
        self.temps += self.dt
        
        # self.temps = round(self.temps, int(1/self.dt))

        # Recompute fields after moving particles
        self.calc_E()
        self.calc_B()

    def create_animation(
        self,
        total_simulation_time,
        total_animation_time,
        animation_type,
        output_folder_name,
        cell_size_reduction,
        v,
        r,
        particule_visualisation,
        min_alpha,
        max_alpha,
    ):

        current_time = datetime.datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Simulation_{current_time_str}.mp4"

        output_folder = os.path.join(
            output_folder_name,
            f"Simulation_{current_time_str}",
        )

        os.makedirs(output_folder, exist_ok=True)
        animation_output_path = os.path.join(output_folder, filename)

        console_output_path = os.path.join(
            output_folder,
            f"Simulation_{current_time_str}_console.txt",
        )

        with open(console_output_path, "w") as console_file:
            sys.stdout = console_file
            print(
                f"Paramètres de simulation : size={self.size}, cell_size={self.cell_size}, dt={self.dt}"
            )
            print(f"Nombre de particules : {len(self.particles)}")
            print("Paramètres initiaux des particules :")
            for i, part in enumerate(self.particles):
                print(
                    f"Particule {i+1}: x={part.x}, y={part.y}, z={part.z}, charge={part.charge}, masse={part.mass}"
                )
            print(f"Type d'animation : {animation_type}")

            print(f"Durée totale de la simulation : {total_simulation_time} s")
            print(f"Durée totale de l'animation : {total_animation_time} s")

            sys.stdout = sys.__stdout__
        fig = plt.figure()
        fig = plt.figure(facecolor="black")
        ax = fig.add_subplot(111, projection="3d")

        ax.view_init(azim=r, elev=v)
        simulation_time = 0  # Initialisation du temps de simulation

        def update():
            nonlocal simulation_time  # Utilisation de la variable simulation_time déclarée en dehors de la fonction

            ax.clear()
            ax.set_xlim(0, self.size - (self.cell_size))
            ax.set_ylim(0, self.size - (self.cell_size))
            ax.set_zlim(0, self.size - (self.cell_size))
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_facecolor("black")

            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.zaxis.label.set_color("white")

            ax.tick_params(axis="x", colors="white")
            ax.tick_params(axis="y", colors="white")
            ax.tick_params(axis="z", colors="white")

            ax.grid(color="white")
            ax.set_title(f"Simulation - Temps écoulé (simulation): {self.temps} s")

            if animation_type == "P":
                self.plot_particle_positions(ax)
            elif animation_type in ["E", "B", "T"]:
                self.plot_fields(
                    ax,
                    field_type=animation_type,
                    cell_size_reduction=cell_size_reduction,
                    min_alpha=min_alpha,
                    max_alpha=max_alpha,
                )
                
            if particule_visualisation == True and animation_type in ["E", "B", "T"]:
                self.plot_fields(
                    ax,
                    field_type=animation_type,
                    cell_size_reduction=cell_size_reduction,
                    min_alpha=min_alpha,
                    max_alpha=max_alpha,
                )
                self.plot_particle_positions(ax)
            self.calc_next()
            ax.set_title(f"Simulation - Temps écoulé (simulation): {self.temps} s", color="white")

        # fps = nombre de frame / temps
        total_animation_frames = int(
            total_simulation_time / self.dt
        )  # Utilisation de self.dt pour le temps d'animation

        # interval (ms) per frame for FuncAnimation
        if total_animation_frames > 0:
            animation_simulation_interval = int((total_animation_time * 1000) / total_animation_frames)
        else:
            animation_simulation_interval = 100

        with tqdm.tqdm(
            total=total_animation_frames, desc="Calcul de l'animation", unit="frames"
        ) as progress_bar:

            def update_with_progress(frame):
                update()
                progress_bar.update(1)

            ani = animation.FuncAnimation(fig, update_with_progress, frames=total_animation_frames, interval=animation_simulation_interval)

            ani.save(animation_output_path, writer="ffmpeg")  # , fps=fps)

            print_debug(f"L'animation a été enregistrée sous {animation_output_path}")
            print_debug(
                f"Les informations sont également enregistrées dans {console_output_path}"
            )
            print_debug(f"{total_animation_time=}")
            print_debug("#######", animation_simulation_interval)

    def plot_particle_positions(self, ax):
            for part in self.particles:
                if part.fil == True:
                    if part == self.particles[0]:
                        color = "r"
                        
                    else:
                        color = "w"
                else:
                    color = "r"

                ax.scatter(part.x, part.y, part.z, c=color, marker="o")
    def solenoide(self,centre_x,centre_y,centre_z,longueur,axe,rayon,densité_de_spires,nombre_total):
        norm_E = (self.I**2)/(self.U*const.epsilon_0)
       
        nombre_de_spire = densité_de_spires*longueur
        if axe == "x":
            for p in range(nombre_total):
                n = ((np.pi*nombre_de_spire) / nombre_total)
                z = centre_x + rayon*np.cos(n*p)
                y = centre_y + rayon*np.sin(n*p)
                x = centre_z + (p/nombre_total ) * longueur
                self.add_part(Particle(x,y,z,const.charge_electron,const.masse_electron,fil = True,solen =True, vx=0.01 , vy = 0.01 ,vz = 0.01))

    
        elif axe == 'y':
            for p in range(nombre_total):
                n = ((np.pi*nombre_de_spire) / nombre_total)
                x = centre_x + rayon*np.cos(n*p)
                z = centre_y + rayon*np.sin(n*p)
                y = centre_z + (p/nombre_total ) * longueur
                self.add_part(Particle(x,y,z,const.charge_electron,const.masse_electron,fil = True,solen =True,vx=0.01 , vy = 0.01, vz = 0.01))

    
        else:
            for p in range(nombre_total):
                
                n = ((np.pi*nombre_de_spire) / nombre_total)
                x = centre_x + rayon*np.cos(n*p)
                y = centre_y + rayon*np.sin(n*p)
                z = centre_z + (p/nombre_total ) * longueur
                
                self.add_part(Particle(x,y,z,const.charge_electron,const.masse_electron,fil = True ,solen =True,vx=0.01 , vy = 0.01 ,vz = 0.01))

            
        
    
    
    
    
    def plot_fields(
            self,
            ax,
            field_type,
            cell_size_reduction,
            min_alpha,
            max_alpha,
        ):

            if field_type == "E":
                field = self.field_E
                field_label = "Champ électrique"
            elif field_type == "B":
                field = self.field_B
                field_label = "Champ magnétique"
            elif field_type == "T":
                field = self.field_E + self.field_B
                field_label = "Champ total (E + B)"
            else:
                raise ValueError("Type de champ non valide. Utilisez 'E', 'B' ou 'TOTAL'.")
           
            shape = field.shape[:-1]
            
            grid_size = np.arange(0, shape[0] * self.cell_size, self.cell_size)
            x_coords, y_coords, z_coords = np.meshgrid(
                grid_size, grid_size, grid_size, indexing="ij"
            )

            # Réduire la taille de la grille
            reduced_shape = (
                shape[0] // cell_size_reduction,
                shape[1] // cell_size_reduction,
                shape[2] // cell_size_reduction,
            )
            x_coords_reduced = x_coords[
                ::cell_size_reduction, ::cell_size_reduction, ::cell_size_reduction
            ]
            y_coords_reduced = y_coords[
                ::cell_size_reduction, ::cell_size_reduction, ::cell_size_reduction
            ]
            z_coords_reduced = z_coords[
                ::cell_size_reduction, ::cell_size_reduction, ::cell_size_reduction
            ]

            # Moyenne des vecteurs de champ dans les cellules
            averaged_field = np.mean(
                field.reshape(
                       (
                        reduced_shape[0],
                        cell_size_reduction,
                        reduced_shape[1],
                        cell_size_reduction,
                        reduced_shape[2],
                        cell_size_reduction,
                        3,
                    )
                ),
                axis=(1, 3, 5),
            )

            # Calcul de la norme du champ moyenné
            norm_values = np.linalg.norm(averaged_field, axis=3)
            norm = plt.Normalize(vmin=norm_values.min(), vmax=norm_values.max())
            norm_values_normalized = norm(norm_values)
            colors = plt.cm.inferno(1 - norm_values_normalized.ravel()**0.5)
            
            alphas = min_alpha + max_alpha * norm_values_normalized.ravel()
            
            # Prefer instance attributes (fall back to module globals)
            axe_local = getattr(self, 'axe', globals().get('axe', 'x'))
            type_simulation_local = getattr(self, 'type_simulation', globals().get('type_simulation', ''))

            if axe_local == "x" and type_simulation_local == "fil":
                angles = np.arctan2(averaged_field[..., 2], averaged_field[..., 0])
                angles_degrees = np.degrees(angles)
                
                angles_normalized = (angles_degrees + 180) / 360  # Normalisation entre 0 et 1
                
                # Utiliser les angles normalisés pour déterminer la couleur de chaque vecteur
                colors = plt.cm.RdYlBu(angles_normalized.ravel())
                
            elif axe_local == "y" and type_simulation_local == "fil":
                angles = np.arctan2(averaged_field[..., 2], averaged_field[..., 1])
                angles_degrees = np.degrees(angles)
                
                angles_normalized = (angles_degrees + 180) / 360  # Normalisation entre 0 et 1
                
                # Utiliser les angles normalisés pour déterminer la couleur de chaque vecteur
                colors = plt.cm.RdYlBu(angles_normalized.ravel())
                
            elif axe_local == "z" and type_simulation_local == "fil":
                angles = np.arctan2(averaged_field[..., 1], averaged_field[..., 0])
                angles_degrees = np.degrees(angles)
                
                angles_normalized = (angles_degrees + 180) / 360  # Normalisation entre 0 et 1
                
                # Utiliser les angles normalisés pour déterminer la couleur de chaque vecteur
                colors = plt.cm.RdYlBu(angles_normalized.ravel())
           
            # Ajuster la taille des flèches pour une meilleure lisibilité
            arrow_length = self.cell_size * cell_size_reduction * 0.08  # Facteur de réduction (0.08 = 8% de la taille de la cellule)
            ax.quiver(
                x_coords_reduced,
                y_coords_reduced,
                z_coords_reduced,
                averaged_field[..., 0],
                averaged_field[..., 1],
                averaged_field[..., 2],
                length=arrow_length,
                normalize=True,
                colors=colors,
                alpha=alphas,
                capstyle='butt',
                arrow_length_ratio=0.5,
            )
            
            
            
            
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.zaxis.label.set_color("white")

            ax.tick_params(axis="x", colors="white")
            ax.tick_params(axis="y", colors="white")
            ax.tick_params(axis="z", colors="white")


    """     
        plt.show()
        
        if not hasattr(self, 'colorbar_created'):
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu, norm=plt.Normalize(vmin=norm_values.min(), vmax=norm_values.max()))
            sm.set_array([])
            self.colorbar = plt.colorbar(sm, ax=ax, label='Norme du champ', pad=0.05)  # Ajustez le pad selon vos besoins
            self.colorbar_created = True"""
        



# ----Temps-----
# The module-level configuration and execution is guarded below so the
# file can be imported safely by tests or other modules without running a
# full simulation on import.

# (Defaults kept for interactive execution when run as a script.)
dt = 0.01  # s (time-step)
duree_simulation = 0.2  # s (total simulation time)
duree_animation = 10  # s

# Flags / modes
clear = False
simulation = True
debug = False
type_simulation = ""  # empty = no filament/random

# Geometry
taille_du_monde = 1.0  # m
taille_des_cellules = 0.05  # m
cell_size_reduction = 2

# Fil parameters
axe = 'x'
position_x = 4  # cell index (0-based)
position_y = 4
I = 40.0
U = 240.0
densité = 1

type_de_courant = "ca"  # "cc" "ca"
f = 100.0  # Hz

# Solenoid / geometry
centre_x = 0.5
centre_y = 0.5
centre_z = 0.0
longueur = 1.0

# Animation defaults
type_aniamtion = "E"  # "P", "E", "B", "T"
particule_visualisation = True

min_alpha = 1.0
max_alpha = 0.0

# View angles
r = 0
v = 0

def crf():
    folder_path = "Résultats"
    if not os.path.isdir(folder_path):
        print_debug(
            f"Le chemin spécifié '{folder_path}' n'est pas un dossier existant."
        )
        return

    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        print_debug(f"Le dossier '{folder_path}' a été vidé avec succès.")
    except Exception as e:
        print_debug(
            f"Erreur lors de la suppression du contenu du dossier '{folder_path}': {e}"
        )


if __name__ == "__main__":
    # Build a World instance from the module-level defaults and run the
    # simulation. This block only executes when the file is run as a script,
    # not when imported.
    w = World(
        taille_du_monde,
        taille_des_cellules,
        dt,
        I=I,
        U=U,
        type_de_courant=type_de_courant,
        f=f,
        axe=axe,
        position_x=position_x,
        position_y=position_y,
        type_simulation=type_simulation,
        centre_x=centre_x,
        centre_y=centre_y,
        centre_z=centre_z,
        longueur=longueur,
        duree_simulation=duree_simulation,
    )

    # Add two large/massive charges placed symmetrically
    center = taille_du_monde / 2
    offset = taille_du_monde * 0.35
    masse_grosse = const.masse_electron * 1e5
    w.add_part(Particle(center - offset, center, center, const.charge_electron, masse_grosse))
    w.add_part(Particle(center + offset, center, center, -const.charge_electron, masse_grosse))

    if type_simulation == "R":
        pass
    elif type_simulation == "fil":
        w.add_fil(axe, position_x, position_y, densité)

    if min_alpha + max_alpha > 1:
        raise ValueError("Alpha ,n'est pas encadré entre 0 et 1")

    if clear:
        crf()

    if simulation:
        # Main FDTD loop
        n_steps = int(duree_simulation / dt)
        for step in range(n_steps):
            w.step_FDTD()
            if step % 10 == 0:
                print(f"Step {step}/{n_steps} : t = {w.temps:.4e} s")

        # Optional animation after the simulation
        w.create_animation(
            duree_simulation,
            duree_animation,
            output_folder_name="Résultats",
            animation_type=type_aniamtion,
            cell_size_reduction=cell_size_reduction,
            r=r,
            v=v,
            particule_visualisation=particule_visualisation,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
        )
