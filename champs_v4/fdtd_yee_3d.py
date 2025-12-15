#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# SenfinaLight, VirinasCode, tt_thoma

"""
Minimal 3D Yee FDTD solver (leap-frog) with support for:
- Staggered fields: Ex, Ey, Ez, Hx, Hy, Hz
- Material maps: epsilon_r (relative permittivity), sigma (conductivity)
- Macroscopic current sources Jx, Jy, Jz
- Optional Numba acceleration (if numba is installed)

This module is intentionally minimal and pedagogical. It is designed to
work in 3D; tests can run in 2D slices by setting one dimension to 1.
"""

import logging
import math
import numpy as np

from .config import int_t, float_t, ndarray_t
from . import methods

import importlib.util

NUMBA = importlib.util.find_spec("numba") is not None
if NUMBA:
    import numba

logger = logging.getLogger(__name__)

# Physical constants
epsilon0 = 8.8541878128e-12
mu0 = 4 * math.pi * 1e-7
c0 = 1.0 / math.sqrt(epsilon0 * mu0)
tiny = 1e-30


class Yee3D:
    """
    3D Finite Difference Time Domain (FDTD) solver using Yee's algorithm.

    This class implements a leap-frog time-stepping scheme for Maxwell's equations
    in 3D Cartesian coordinates. Fields are staggered on a Yee grid:
    - Electric fields (Ex, Ey, Ez) are located at edges of cells.
    - Magnetic fields (Hx, Hy, Hz) are located at faces of cells.

    Supports:
    - Material properties: relative permittivity (epsilon_r) and conductivity (sigma).
    - Macroscopic current sources (Jx, Jy, Jz).
    - Perfectly Matched Layer (PML) absorption boundaries.
    - Optional Numba JIT compilation for performance.

    Attributes:
        nx, ny, nz (int): Number of cells in x, y, z directions.
        dx (float): Spatial step size (meters).
        dt (float): Time step size (seconds).
        Ex, Ey, Ez (np.ndarray): Electric field components.
        Hx, Hy, Hz (np.ndarray): Magnetic field components.
        epsilon_r (np.ndarray): Relative permittivity map (shape: nx, ny, nz).
        sigma (np.ndarray): Conductivity map (shape: nx, ny, nz).
        Jx, Jy, Jz (np.ndarray): Current density sources.

    Parameters for __init__:
        nx, ny, nz: Grid dimensions.
        dx: Spatial resolution.
        dt: Time step (must satisfy CFL condition: dt < dx / c0 / sqrt(3)).
        use_numba: Enable Numba acceleration if available.
        pml_width: Width of PML boundary layer.
        pml_sigma_max: Maximum conductivity in PML.
    """
    def __init__(
            self,
            nx: int, ny: int, nz: int, dx: float, dt: float,
            *, pml_width: int = 10, pml_sigma_max: float = 1.0
    ):
        """
        Initialize a 3D Yee grid.
        nx,ny,nz : number of cells in each direction (integer)
        dx : spatial step (assumed equal in all directions)
        dt : time step (should satisfy CFL)
        """
        self.range = range

        self.nx: int_t = int_t(nx)
        self.ny: int_t = int_t(ny)
        self.nz: int_t = int_t(nz)

        self.dx: float_t = float_t(dx)
        self.dt: float_t = float_t(dt)

        self.psi_ex_dy = None
        self.psi_ex_dz = None

        self.psi_ey_dx = None
        self.psi_ey_dz = None

        self.psi_ez_dx = None
        self.psi_ez_dy = None

        self.psi_hx_dy = None

        self.psi_hy_dx = None
        self.psi_hy_dz = None

        self.psi_hz_dy = None

        # fields: using Yee stagger: E components centered on edges,
        # H components centered on faces. We'll store arrays sized so
        # that finite difference indexing is straightforward.
        self.Ex: ndarray_t = np.zeros((nx    , ny + 1, nz + 1), dtype=float_t)
        self.Ey: ndarray_t = np.zeros((nx + 1, ny    , nz + 1), dtype=float_t)
        self.Ez: ndarray_t = np.zeros((nx + 1, ny + 1, nz    ), dtype=float_t)

        self.Hx: ndarray_t = np.zeros((nx + 1, ny    , nz    ), dtype=float_t)
        self.Hy: ndarray_t = np.zeros((nx    , ny + 1, nz    ), dtype=float_t)
        self.Hz: ndarray_t = np.zeros((nx    , ny    , nz + 1), dtype=float_t)

        self.epsilon_Ex = None
        self.epsilon_Ey = None
        self.epsilon_Ez = None

        self.sigma_Ex = None
        self.sigma_Ey = None
        self.sigma_Ez = None

        self.cex = None
        self.cey = None
        self.cez = None

        # material maps (cell-centered)
        self.epsilon_r: ndarray_t = np.ones((nx, ny, nz), dtype=float_t)
        self.sigma: ndarray_t = np.zeros((nx, ny, nz), dtype=float_t)

        # macroscopic current density on staggered grid corresponding to E
        self.Jx = np.zeros_like(self.Ex)
        self.Jy = np.zeros_like(self.Ey)
        self.Jz = np.zeros_like(self.Ez)

        # Derived coefficients for update (will be computed)
        self._compute_material_coefficients()

        # PML (simple exponential damping mask) parameters
        self.pml_width = int_t(pml_width)
        self.pml_sigma_max = float_t(pml_sigma_max)
        self._init_pml()

    def _compute_material_coefficients(self):
        (
            self.epsilon_Ex, self.epsilon_Ey, self.epsilon_Ez,
            self.sigma_Ex, self.sigma_Ey, self.sigma_Ez,
            self.cex, self.cey, self.cez
        ) = methods.compute_material_coefficients(
            self.nx, self.ny, self.nz,
            self.Ex, self.Ey, self.Ez,
            self.epsilon_r, self.sigma,
            self.dt
        )

    def set_materials(self, epsilon_r: np.ndarray, sigma: ndarray_t = None):
        """
        Set material properties for the grid.

        Parameters:
            epsilon_r (np.ndarray): Relative permittivity map, shape (nx, ny, nz).
            sigma (np.ndarray, optional): Conductivity map, shape (nx, ny, nz). Defaults to zeros.

        Raises:
            AssertionError: If array shapes do not match (nx, ny, nz).
        """
        self.epsilon_r, sigma = methods.set_materials(
            self.nx, self.ny, self.nz,
            epsilon_r, sigma
        )

        self.sigma = sigma or self.sigma
        self._compute_material_coefficients()

    def add_coil(self,
                 center: tuple[int_t, int_t, int_t], radius_cells,
                 *, axis: str = 'z', turns: int_t = 1, current: float_t = 1.0
        ):
        """
        Approximate a coil (solenoid) by adding macroscopic current density J
        along axis direction distributed over the loop cells.

        center : (ix,iy,iz) cell indices for coil center
        radius_cells : radius in cells
        axis : 'x','y' or 'z'
        turns : number of turns (multiplicative factor)
        current : current amplitude (A)
        """

        self.Jz = methods.add_coil(
            self.nx, self.ny, self.nz,
            self.Ez, self.Jz,
            center, radius_cells,
            axis=axis, turns=turns, current=current
        )

    def step(self):
        """
        Advance the electromagnetic fields by one time step using the FDTD leap-frog scheme.

        Updates magnetic fields (H) first, then electric fields (E). Applies PML damping
        to both field types if configured. Current sources (J) are incorporated in E updates.
        """
        self.update_H()

        # apply simple H damping in PML
        if self.dampH_Hx is not None:
            self.Hx *= self.dampH_Hx
        if self.dampH_Hy  is not None:
            self.Hy *= self.dampH_Hy
        if self.dampH_Hz is not None:
            self.Hz *= self.dampH_Hz

        self.update_E()
        # apply simple E damping in PML
        if self.dampE is not None:
            # Ex/Ey/Ez have different shapes; apply component masks if available
            if self.dampE_Ex is not None:
                self.Ex *= self.dampE_Ex
            else:
                self.Ex *= self.dampE

            if self.dampE_Ey is not None:
                self.Ey *= self.dampE_Ey
            else:
                self.Ey *= self.dampE

            if self.dampE_Ez is not None:
                self.Ez *= self.dampE_Ez
            else:
                self.Ez *= self.dampE

    def update_H(self):
        (
            self.psi_ex_dy, self.psi_ex_dz,
            self.psi_ey_dx, self.psi_ey_dz,
            self.psi_ez_dx, self.psi_ez_dy,
            self.Hx, self.Hy, self.Hz
        ) = methods.update_H(
            self.Ex, self.Ey, self.Ez,
            self.Hx, self.Hy, self.Hz,

            self.psi_ex_dy, self.psi_ex_dz,
            self.psi_ey_dx, self.psi_ey_dz,
            self.psi_ez_dx, self.psi_ez_dy,

            self.b_ex_dy, self.c_ex_dy,
            self.b_ex_dz, self.c_ex_dz,

            self.b_ey_dx, self.c_ey_dx,
            self.b_ey_dz, self.c_ey_dz,

            self.b_ez_dx, self.c_ez_dx,
            self.b_ez_dy, self.c_ez_dy,

            self.dt, self.dx
        )

    def update_E(self):
        (
            self.psi_hx_dy, self.psi_hx_dz,
            self.psi_hy_dx, self.psi_hy_dz,
            self.psi_hz_dx, self.psi_hz_dy,
            self.Ex, self.Ey, self.Ez
        ) = methods.update_E(
            self.nx, self.ny, self.nz,

            self.epsilon_Ex, self.epsilon_Ey, self.epsilon_Ez,

            self.sigma_Ex, self.sigma_Ey, self.sigma_Ez,

            self.Ex, self.Ey, self.Ez,
            self.Hx, self.Hy, self.Hz,
            self.Jx, self.Jy, self.Jz,

            self.b_hx_dy, self.c_hx_dy,
            self.b_hx_dz, self.c_hx_dz,

            self.b_hy_dx, self.c_hy_dx,
            self.b_hy_dz, self.c_hy_dz,

            self.b_hz_dx, self.c_hz_dx,
            self.b_hz_dy, self.c_hz_dy,

            self.dt, self.dx
        )

    def _init_pml(self):
        """
        Initialize simple exponential damping masks for E and H fields.
        This is a lightweight PML substitute: it applies an exponential
        damping mask near boundaries to absorb outgoing waves. Not a full
        CPML implementation, but effective for reducing reflections.
        """

        w = max(0, int_t(self.pml_width))
        sigma_max = float_t(self.pml_sigma_max)

        def axis_sigma(L):
            s = np.zeros(L, dtype=float_t)
            if w <= 0:
                return s
            # limit local width to at most half the axis length
            w_local = min(w, L // 2)
            for idx in self.range(w_local):
                frac = (w_local - idx) / float_t(max(1, w_local))
                val = (frac**2) * sigma_max
                s[idx] = val
                s[-1 - idx] = val
            return s

        dt = self.dt
        # Ex shape: (nx, ny+1, nz+1)
        sx = axis_sigma(self.Ex.shape[0])[:, None, None]
        sy = axis_sigma(self.Ex.shape[1])[None, :, None]
        sz = axis_sigma(self.Ex.shape[2])[None, None, :]
        sigma_total_Ex = sx + sy + sz

        # Ey shape: (nx+1, ny, nz+1)
        sx2 = axis_sigma(self.Ey.shape[0])[:, None, None]
        sy2 = axis_sigma(self.Ey.shape[1])[None, :, None]
        sz2 = axis_sigma(self.Ey.shape[2])[None, None, :]
        sigma_total_Ey = sx2 + sy2 + sz2

        # Ez shape: (nx+1, ny+1, nz)
        sx3 = axis_sigma(self.Ez.shape[0])[:, None, None]
        sy3 = axis_sigma(self.Ez.shape[1])[None, :, None]
        sz3 = axis_sigma(self.Ez.shape[2])[None, None, :]
        sigma_total_Ez = sx3 + sy3 + sz3

        # H shapes: compute masks per H-component shape
        sx_hx = axis_sigma(self.Hx.shape[0])[:, None, None]
        sy_hx = axis_sigma(self.Hx.shape[1])[None, :, None]
        sz_hx = axis_sigma(self.Hx.shape[2])[None, None, :]
        sigma_total_Hx = sx_hx + sy_hx + sz_hx

        sx_hy = axis_sigma(self.Hy.shape[0])[:, None, None]
        sy_hy = axis_sigma(self.Hy.shape[1])[None, :, None]
        sz_hy = axis_sigma(self.Hy.shape[2])[None, None, :]
        sigma_total_Hy = sx_hy + sy_hy + sz_hy

        sx_hz = axis_sigma(self.Hz.shape[0])[:, None, None]
        sy_hz = axis_sigma(self.Hz.shape[1])[None, :, None]
        sz_hz = axis_sigma(self.Hz.shape[2])[None, None, :]
        sigma_total_Hz = sx_hz + sy_hz + sz_hz

        with np.errstate(over='ignore'):
            self.dampE_Ex = np.exp(-sigma_total_Ex * dt / epsilon0)
            self.dampE_Ey = np.exp(-sigma_total_Ey * dt / epsilon0)
            self.dampE_Ez = np.exp(-sigma_total_Ez * dt / epsilon0)
            self.dampH_Hx = np.exp(-sigma_total_Hx * dt / mu0)
            self.dampH_Hy = np.exp(-sigma_total_Hy * dt / mu0)
            self.dampH_Hz = np.exp(-sigma_total_Hz * dt / mu0)

        # Backwards-compatible single-name mask for Ex
        self.dampE = self.dampE_Ex

        # --- CPML auxiliary arrays (psi) and recursion coefficients ---
        # We will build psi arrays after computing sigma arrays for each
        # derivative location (below). This ensures psi shapes match the
        # actual derivative arrays used in updates.
        # Helper to average sigma_total arrays to derivative locations
        # For derivative shapes we average neighboring sigma values
        # sigma for dEz/dy (Ez shape: (nx+1, ny+1, nz)) -> average at Ez[:-1,1:] and Ez[:-1,:-1]
        sigma_ex_dy = 0.5 * (sigma_total_Ex[:, 1:, :-1] + sigma_total_Ex[:, :-1, :-1])
        sigma_ex_dz = 0.5 * (sigma_total_Ex[:, :-1, 1:] + sigma_total_Ex[:, :-1, :-1])

        sigma_ey_dx = 0.5 * (sigma_total_Ey[1:, :, :-1] + sigma_total_Ey[:-1, :, :-1])
        sigma_ey_dz = 0.5 * (sigma_total_Ey[:-1, :, 1:] + sigma_total_Ey[:-1, :, :-1])

        sigma_ez_dx = 0.5 * (sigma_total_Ez[1:, :-1, :] + sigma_total_Ez[:-1, :-1, :])
        sigma_ez_dy = 0.5 * (sigma_total_Ez[:-1, 1:, :] + sigma_total_Ez[:-1, :-1, :])

        # For derivatives of H, use H sigma_total analogs (sigma_total_Hx etc.)
        # Use the exact same slices as the derivatives in update_E to build sigma
        sigma_hx_dy = 0.5 * (sigma_total_Hx[1:self.nx, 1:self.ny, 0:self.nz] + sigma_total_Hx[1:self.nx, 0:self.ny-1, 0:self.nz])
        sigma_hx_dz = 0.5 * (sigma_total_Hx[1:self.nx, 0:self.ny, 1:self.nz] + sigma_total_Hx[1:self.nx, 0:self.ny, 0:self.nz-1])

        sigma_hy_dx = 0.5 * (sigma_total_Hy[1:self.nx, 1:self.ny, 0:self.nz] + sigma_total_Hy[0:self.nx-1, 1:self.ny, 0:self.nz])
        sigma_hy_dz = 0.5 * (sigma_total_Hy[0:self.nx, 1:self.ny, 1:self.nz] + sigma_total_Hy[0:self.nx, 1:self.ny, 0:self.nz-1])

        sigma_hz_dx = 0.5 * (sigma_total_Hz[1:self.nx, 0:self.ny, 1:self.nz] + sigma_total_Hz[0:self.nx-1, 0:self.ny, 1:self.nz])
        sigma_hz_dy = 0.5 * (sigma_total_Hz[0:self.nx, 1:self.ny, 1:self.nz] + sigma_total_Hz[0:self.nx, 0:self.ny-1, 1:self.nz])

        # Ensure shapes equal to (nx,ny,nz)
        # Recursion coefficients b = exp(-sigma*dt/param), c = (1-b)/sigma (approx)
        b_scale_e = dt / epsilon0
        b_scale_h = dt / mu0

        def make_bc(sigma_arr, scale):
            b = np.exp(-sigma_arr * scale)
            # c: avoid divide by zero: if sigma small, c ~ scale (i.e., dt/eps)
            with np.errstate(divide='ignore', invalid='ignore'):
                c = np.where(sigma_arr > 0, (1.0 - b) / (sigma_arr + tiny), scale)
            return b, c

        self.b_ex_dy, self.c_ex_dy = make_bc(sigma_ex_dy, b_scale_e)
        self.b_ex_dz, self.c_ex_dz = make_bc(sigma_ex_dz, b_scale_e)

        self.b_ey_dx, self.c_ey_dx = make_bc(sigma_ey_dx, b_scale_e)
        self.b_ey_dz, self.c_ey_dz = make_bc(sigma_ey_dz, b_scale_e)

        self.b_ez_dx, self.c_ez_dx = make_bc(sigma_ez_dx, b_scale_e)
        self.b_ez_dy, self.c_ez_dy = make_bc(sigma_ez_dy, b_scale_e)

        self.b_hx_dy, self.c_hx_dy = make_bc(sigma_hx_dy, b_scale_h)
        self.b_hx_dz, self.c_hx_dz = make_bc(sigma_hx_dz, b_scale_h)

        self.b_hy_dx, self.c_hy_dx = make_bc(sigma_hy_dx, b_scale_h)
        self.b_hy_dz, self.c_hy_dz = make_bc(sigma_hy_dz, b_scale_h)

        self.b_hz_dx, self.c_hz_dx = make_bc(sigma_hz_dx, b_scale_h)
        self.b_hz_dy, self.c_hz_dy = make_bc(sigma_hz_dy, b_scale_h)

        # Initialize psi arrays with shapes matching sigma derivative arrays
        self.psi_ex_dy = np.zeros_like(sigma_ex_dy)
        self.psi_ex_dz = np.zeros_like(sigma_ex_dz)

        self.psi_ey_dx = np.zeros_like(sigma_ey_dx)
        self.psi_ey_dz = np.zeros_like(sigma_ey_dz)

        self.psi_ez_dx = np.zeros_like(sigma_ez_dx)
        self.psi_ez_dy = np.zeros_like(sigma_ez_dy)

        self.psi_hx_dy = np.zeros_like(sigma_hx_dy)
        self.psi_hx_dz = np.zeros_like(sigma_hx_dz)

        self.psi_hy_dx = np.zeros_like(sigma_hy_dx)
        self.psi_hy_dz = np.zeros_like(sigma_hy_dz)

        self.psi_hz_dx = np.zeros_like(sigma_hz_dx)
        self.psi_hz_dy = np.zeros_like(sigma_hz_dy)

if __name__ == '__main__':
    print('Yee3D module loaded. Use from tests or scripts.')
