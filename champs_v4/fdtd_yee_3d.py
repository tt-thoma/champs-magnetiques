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
        logger.info("Initializing Yee3D instance")
        self.range = range

        self.nx: int_t = int_t(nx)
        self.ny: int_t = int_t(ny)
        self.nz: int_t = int_t(nz)

        self.dx: float_t = float_t(dx)
        self.dt: float_t = float_t(dt)
        
        # Time tracking
        self.current_time: float_t = 0.0
        self.step_count: int = 0
        
        # Probe manager (optional)
        self.probe_manager = None

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
        logger.info("Finished initializing instance")

    def _compute_material_coefficients(self):
        logger.info("Computing materials")
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
        logger.info("Finished computing materials")

    def set_materials(self, epsilon_r: np.ndarray, sigma: ndarray_t = None):
        """
        Set material properties for the grid.

        Parameters:
            epsilon_r (np.ndarray): Relative permittivity map, shape (nx, ny, nz).
            sigma (np.ndarray, optional): Conductivity map, shape (nx, ny, nz). Defaults to zeros.

        Raises:
            AssertionError: If array shapes do not match (nx, ny, nz).
        """
        logger.info("Setting material")
        assert epsilon_r.shape == (self.nx, self.ny, self.nz)
        if sigma is not None:
            assert sigma.shape == (self.nx, self.ny, self.nz)

        self.epsilon_r, sigma = methods.set_materials(
            self.nx, self.ny, self.nz,
            epsilon_r, sigma
        )

        self.sigma = self.sigma if sigma is None else sigma
        self._compute_material_coefficients()
        logger.info("Finished setting material")

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
        logger.info("Adding coil")
        assert axis in ("x", "y", "z")
        if axis != "z":
            raise NotImplementedError

        self.Jz = methods.add_coil(
            self.nx, self.ny, self.nz,
            self.Ez, self.Jz,
            center, radius_cells,
            axis=axis, turns=turns, current=current
        )
        logger.info("Finished adding coil")

    def step(self):
        """
        Advance the electromagnetic fields by one time step using the FDTD leap-frog scheme.

        Updates magnetic fields (H) first, then electric fields (E). Applies PML damping
        to both field types if configured. Current sources (J) are incorporated in E updates.
        """
        (
            self.Hx, self.Hy, self.Hz,
            self.Ex, self.Ey, self.Ez,

            self.psi_ex_dy, self.psi_ex_dz,
            self.psi_ey_dx, self.psi_ey_dz,
            self.psi_ez_dx, self.psi_ez_dy,

            self.psi_hx_dy, self.psi_hx_dz,
            self.psi_hy_dx, self.psi_hy_dz,
            self.psi_hz_dx, self.psi_hz_dy
        ) = methods.step(
            self.nx, self.ny, self.nz,

            self.Ex, self.Ey, self.Ez,
            self.Hx, self.Hy, self.Hz,
            self.Jx, self.Jy, self.Jz,

            self.dampE, self.dampE_Ex, self.dampE_Ey, self.dampE_Ez,

            self.epsilon_Ex, self.epsilon_Ey, self.epsilon_Ez,
            self.sigma_Ex, self.sigma_Ey, self.sigma_Ez,

            self.psi_ex_dy, self.psi_ex_dz,
            self.psi_ey_dx, self.psi_ey_dz,
            self.psi_ez_dx, self.psi_ez_dy,

            self.b_ex_dy, self.c_ex_dy, self.kappa_ex_dy,
            self.b_ex_dz, self.c_ex_dz, self.kappa_ex_dz,

            self.b_ey_dx, self.c_ey_dx, self.kappa_ey_dx,
            self.b_ey_dz, self.c_ey_dz, self.kappa_ey_dz,

            self.b_ez_dx, self.c_ez_dx, self.kappa_ez_dx,
            self.b_ez_dy, self.c_ez_dy, self.kappa_ez_dy,

            self.psi_hx_dy, self.psi_hx_dz,
            self.psi_hy_dx, self.psi_hy_dz,
            self.psi_hz_dx, self.psi_hz_dy,

            self.b_hx_dy, self.c_hx_dy, self.kappa_hx_dy,
            self.b_hx_dz, self.c_hx_dz, self.kappa_hx_dz,

            self.b_hy_dx, self.c_hy_dx, self.kappa_hy_dx,
            self.b_hy_dz, self.c_hy_dz, self.kappa_hy_dz,

            self.b_hz_dx, self.c_hz_dx, self.kappa_hz_dx,
            self.b_hz_dy, self.c_hz_dy, self.kappa_hz_dy,

            self.dampH_Hx, self.dampH_Hy, self.dampH_Hz,

            self.dt, self.dx
        )
        
        # Update time tracking
        self.step_count += 1
        self.current_time += self.dt
        
        # Sample probes if manager exists
        if self.probe_manager is not None:
            self.probe_manager.sample_all(self, self.current_time)

    def _init_pml(self):
        """
        Initialize simple exponential damping masks for E and H fields.
        This is a lightweight PML substitute: it applies an exponential
        damping mask near boundaries to absorb outgoing waves. Not a full
        CPML implementation, but effective for reducing reflections.
        """
        logger.info("Initializing PML")
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
        self.kappa_ex_dy = np.ones_like(sigma_ex_dy)
        self.b_ex_dz, self.c_ex_dz = make_bc(sigma_ex_dz, b_scale_e)
        self.kappa_ex_dz = np.ones_like(sigma_ex_dz)

        self.b_ey_dx, self.c_ey_dx = make_bc(sigma_ey_dx, b_scale_e)
        self.kappa_ey_dx = np.ones_like(sigma_ey_dx)
        self.b_ey_dz, self.c_ey_dz = make_bc(sigma_ey_dz, b_scale_e)
        self.kappa_ey_dz = np.ones_like(sigma_ey_dz)

        self.b_ez_dx, self.c_ez_dx = make_bc(sigma_ez_dx, b_scale_e)
        self.kappa_ez_dx = np.ones_like(sigma_ez_dx)
        self.b_ez_dy, self.c_ez_dy = make_bc(sigma_ez_dy, b_scale_e)
        self.kappa_ez_dy = np.ones_like(sigma_ez_dy)

        self.b_hx_dy, self.c_hx_dy = make_bc(sigma_hx_dy, b_scale_h)
        self.kappa_hx_dy = np.ones_like(sigma_hx_dy)
        self.b_hx_dz, self.c_hx_dz = make_bc(sigma_hx_dz, b_scale_h)
        self.kappa_hx_dz = np.ones_like(sigma_hx_dz)

        self.b_hy_dx, self.c_hy_dx = make_bc(sigma_hy_dx, b_scale_h)
        self.kappa_hy_dx = np.ones_like(sigma_hy_dx)
        self.b_hy_dz, self.c_hy_dz = make_bc(sigma_hy_dz, b_scale_h)
        self.kappa_hy_dz = np.ones_like(sigma_hy_dz)

        self.b_hz_dx, self.c_hz_dx = make_bc(sigma_hz_dx, b_scale_h)
        self.kappa_hz_dx = np.ones_like(sigma_hz_dx)
        self.b_hz_dy, self.c_hz_dy = make_bc(sigma_hz_dy, b_scale_h)
        self.kappa_hz_dy = np.ones_like(sigma_hz_dy)

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
        logger.info("Finished initializing PML")

    def compute_energy(self):
        """
        Compute the total electromagnetic energy in the simulation domain.
        
        Returns the electric and magnetic energy densities integrated over the volume,
        taking into account the Yee grid staggering by interpolating fields to cell centers.
        
        Energy formulas:
        - Electric energy: U_E = (1/2) * ε₀ * εᵣ * |E|² * dV
        - Magnetic energy: U_H = (1/2) * μ₀ * |H|² * dV
        
        Returns:
            tuple: (U_electric, U_magnetic, U_total) in Joules
        
        Note:
            The fields are interpolated to cell centers to properly account for
            the staggered Yee grid geometry. This ensures energy is evaluated
            at consistent spatial locations.
        """
        # Volume element
        dV = self.dx**3
        
        # Interpolate E components to cell centers (nx, ny, nz)
        # Ex is at (i, j+0.5, k+0.5), average to get at (i, j, k)
        Ex_center = 0.25 * (
            self.Ex[:self.nx, :self.ny, :self.nz] +
            self.Ex[:self.nx, 1:self.ny+1, :self.nz] +
            self.Ex[:self.nx, :self.ny, 1:self.nz+1] +
            self.Ex[:self.nx, 1:self.ny+1, 1:self.nz+1]
        )
        
        # Ey is at (i+0.5, j, k+0.5), average to get at (i, j, k)
        Ey_center = 0.25 * (
            self.Ey[:self.nx, :self.ny, :self.nz] +
            self.Ey[1:self.nx+1, :self.ny, :self.nz] +
            self.Ey[:self.nx, :self.ny, 1:self.nz+1] +
            self.Ey[1:self.nx+1, :self.ny, 1:self.nz+1]
        )
        
        # Ez is at (i+0.5, j+0.5, k), average to get at (i, j, k)
        Ez_center = 0.25 * (
            self.Ez[:self.nx, :self.ny, :self.nz] +
            self.Ez[1:self.nx+1, :self.ny, :self.nz] +
            self.Ez[:self.nx, 1:self.ny+1, :self.nz] +
            self.Ez[1:self.nx+1, 1:self.ny+1, :self.nz]
        )
        
        # Interpolate H components to cell centers (nx, ny, nz)
        # Hx is at (i+0.5, j, k), average to get at (i, j, k)
        Hx_center = 0.5 * (
            self.Hx[:self.nx, :self.ny, :self.nz] +
            self.Hx[1:self.nx+1, :self.ny, :self.nz]
        )
        
        # Hy is at (i, j+0.5, k), average to get at (i, j, k)
        Hy_center = 0.5 * (
            self.Hy[:self.nx, :self.ny, :self.nz] +
            self.Hy[:self.nx, 1:self.ny+1, :self.nz]
        )
        
        # Hz is at (i, j, k+0.5), average to get at (i, j, k)
        Hz_center = 0.5 * (
            self.Hz[:self.nx, :self.ny, :self.nz] +
            self.Hz[:self.nx, :self.ny, 1:self.nz+1]
        )
        
        # Electric field magnitude squared at cell centers
        E2 = Ex_center**2 + Ey_center**2 + Ez_center**2
        
        # Magnetic field magnitude squared at cell centers
        H2 = Hx_center**2 + Hy_center**2 + Hz_center**2
        
        # Electric energy density: (1/2) * ε₀ * εᵣ * |E|²
        # Note: epsilon_r is already defined at cell centers (nx, ny, nz)
        u_electric = 0.5 * epsilon0 * self.epsilon_r * E2
        
        # Magnetic energy density: (1/2) * μ₀ * |H|²
        u_magnetic = 0.5 * mu0 * H2
        
        # Integrate over volume
        U_electric = np.sum(u_electric) * dV
        U_magnetic = np.sum(u_magnetic) * dV
        U_total = U_electric + U_magnetic
        
        return U_electric, U_magnetic, U_total

    def compute_energy_in_region(self, x_slice=None, y_slice=None, z_slice=None):
        """
        Compute electromagnetic energy in a specific region of the domain.
        
        Parameters:
            x_slice (slice, optional): Region in x-direction (e.g., slice(10, 50))
            y_slice (slice, optional): Region in y-direction
            z_slice (slice, optional): Region in z-direction
            
        Returns:
            tuple: (U_electric, U_magnetic, U_total) in the specified region
            
        Example:
            # Energy in central 50% of domain
            U_e, U_m, U_tot = sim.compute_energy_in_region(
                slice(nx//4, 3*nx//4),
                slice(ny//4, 3*ny//4),
                slice(nz//4, 3*nz//4)
            )
        """
        # Default to full domain
        x_slice = x_slice or slice(None)
        y_slice = y_slice or slice(None)
        z_slice = z_slice or slice(None)
        
        # Volume element
        dV = self.dx**3
        
        # Interpolate E components to cell centers with slicing
        Ex_center = 0.25 * (
            self.Ex[x_slice, y_slice, z_slice] +
            self.Ex[x_slice, slice((y_slice.start or 0) + 1, (y_slice.stop or self.ny) + 1), z_slice] +
            self.Ex[x_slice, y_slice, slice((z_slice.start or 0) + 1, (z_slice.stop or self.nz) + 1)] +
            self.Ex[x_slice, slice((y_slice.start or 0) + 1, (y_slice.stop or self.ny) + 1), 
                    slice((z_slice.start or 0) + 1, (z_slice.stop or self.nz) + 1)]
        )
        
        Ey_center = 0.25 * (
            self.Ey[x_slice, y_slice, z_slice] +
            self.Ey[slice((x_slice.start or 0) + 1, (x_slice.stop or self.nx) + 1), y_slice, z_slice] +
            self.Ey[x_slice, y_slice, slice((z_slice.start or 0) + 1, (z_slice.stop or self.nz) + 1)] +
            self.Ey[slice((x_slice.start or 0) + 1, (x_slice.stop or self.nx) + 1), y_slice, 
                    slice((z_slice.start or 0) + 1, (z_slice.stop or self.nz) + 1)]
        )
        
        Ez_center = 0.25 * (
            self.Ez[x_slice, y_slice, z_slice] +
            self.Ez[slice((x_slice.start or 0) + 1, (x_slice.stop or self.nx) + 1), y_slice, z_slice] +
            self.Ez[x_slice, slice((y_slice.start or 0) + 1, (y_slice.stop or self.ny) + 1), z_slice] +
            self.Ez[slice((x_slice.start or 0) + 1, (x_slice.stop or self.nx) + 1), 
                    slice((y_slice.start or 0) + 1, (y_slice.stop or self.ny) + 1), z_slice]
        )
        
        # Interpolate H components to cell centers with slicing
        Hx_center = 0.5 * (
            self.Hx[x_slice, y_slice, z_slice] +
            self.Hx[slice((x_slice.start or 0) + 1, (x_slice.stop or self.nx) + 1), y_slice, z_slice]
        )
        
        Hy_center = 0.5 * (
            self.Hy[x_slice, y_slice, z_slice] +
            self.Hy[x_slice, slice((y_slice.start or 0) + 1, (y_slice.stop or self.ny) + 1), z_slice]
        )
        
        Hz_center = 0.5 * (
            self.Hz[x_slice, y_slice, z_slice] +
            self.Hz[x_slice, y_slice, slice((z_slice.start or 0) + 1, (z_slice.stop or self.nz) + 1)]
        )
        
        # Field magnitudes squared
        E2 = Ex_center**2 + Ey_center**2 + Ez_center**2
        H2 = Hx_center**2 + Hy_center**2 + Hz_center**2
        
        # Energy densities
        epsilon_r_region = self.epsilon_r[x_slice, y_slice, z_slice]
        u_electric = 0.5 * epsilon0 * epsilon_r_region * E2
        u_magnetic = 0.5 * mu0 * H2
        
        # Integrate
        U_electric = np.sum(u_electric) * dV
        U_magnetic = np.sum(u_magnetic) * dV
        U_total = U_electric + U_magnetic
        
        return U_electric, U_magnetic, U_total

    def compute_poynting_vector(self):
        """
        Compute the Poynting vector S = E × H at cell centers.
        
        The Poynting vector represents the directional energy flux (power per unit area)
        of electromagnetic fields. It points in the direction of wave propagation.
        
        Formula: S = E × H (in SI units, W/m²)
        
        Returns:
            tuple: (Sx, Sy, Sz) - Components of Poynting vector at cell centers (nx, ny, nz)
                   Each component is in W/m²
        
        Note:
            Fields are interpolated to cell centers to account for Yee grid staggering.
            The cross product is computed as:
            - Sx = Ey*Hz - Ez*Hy
            - Sy = Ez*Hx - Ex*Hz
            - Sz = Ex*Hy - Ey*Hx
        """
        # Interpolate E components to cell centers (nx, ny, nz)
        Ex_center = 0.25 * (
            self.Ex[:self.nx, :self.ny, :self.nz] +
            self.Ex[:self.nx, 1:self.ny+1, :self.nz] +
            self.Ex[:self.nx, :self.ny, 1:self.nz+1] +
            self.Ex[:self.nx, 1:self.ny+1, 1:self.nz+1]
        )
        
        Ey_center = 0.25 * (
            self.Ey[:self.nx, :self.ny, :self.nz] +
            self.Ey[1:self.nx+1, :self.ny, :self.nz] +
            self.Ey[:self.nx, :self.ny, 1:self.nz+1] +
            self.Ey[1:self.nx+1, :self.ny, 1:self.nz+1]
        )
        
        Ez_center = 0.25 * (
            self.Ez[:self.nx, :self.ny, :self.nz] +
            self.Ez[1:self.nx+1, :self.ny, :self.nz] +
            self.Ez[:self.nx, 1:self.ny+1, :self.nz] +
            self.Ez[1:self.nx+1, 1:self.ny+1, :self.nz]
        )
        
        # Interpolate H components to cell centers (nx, ny, nz)
        Hx_center = 0.5 * (
            self.Hx[:self.nx, :self.ny, :self.nz] +
            self.Hx[1:self.nx+1, :self.ny, :self.nz]
        )
        
        Hy_center = 0.5 * (
            self.Hy[:self.nx, :self.ny, :self.nz] +
            self.Hy[:self.nx, 1:self.ny+1, :self.nz]
        )
        
        Hz_center = 0.5 * (
            self.Hz[:self.nx, :self.ny, :self.nz] +
            self.Hz[:self.nx, :self.ny, 1:self.nz+1]
        )
        
        # Compute cross product: S = E × H
        Sx = Ey_center * Hz_center - Ez_center * Hy_center
        Sy = Ez_center * Hx_center - Ex_center * Hz_center
        Sz = Ex_center * Hy_center - Ey_center * Hx_center
        
        return Sx, Sy, Sz

    def compute_poynting_flux(self, axis='z', position=None):
        """
        Compute the total power flux through a plane perpendicular to a given axis.
        
        This calculates the integral of the Poynting vector component normal to a plane,
        representing the total electromagnetic power passing through that plane.
        
        Parameters:
            axis (str): Normal axis to the plane ('x', 'y', or 'z')
            position (int, optional): Position along axis (cell index). 
                                     Defaults to middle of domain.
        
        Returns:
            float: Total power flux through the plane (Watts)
        
        Example:
            # Power flux through z-plane at middle
            power = sim.compute_poynting_flux(axis='z', position=nz//2)
            
            # Power entering/leaving through boundaries
            power_left = sim.compute_poynting_flux('x', 0)
            power_right = sim.compute_poynting_flux('x', nx-1)
        """
        Sx, Sy, Sz = self.compute_poynting_vector()
        
        # Area element perpendicular to axis
        dA = self.dx**2
        
        if axis == 'x':
            pos = position if position is not None else self.nx // 2
            # Flux through yz-plane at x=pos
            flux = np.sum(Sx[pos, :, :]) * dA
        elif axis == 'y':
            pos = position if position is not None else self.ny // 2
            # Flux through xz-plane at y=pos
            flux = np.sum(Sy[:, pos, :]) * dA
        elif axis == 'z':
            pos = position if position is not None else self.nz // 2
            # Flux through xy-plane at z=pos
            flux = np.sum(Sz[:, :, pos]) * dA
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")
        
        return flux

    def compute_poynting_flux_boundaries(self):
        """
        Compute net power flux through all six domain boundaries.
        
        Returns:
            dict: Power flux through each boundary (Watts)
                 Keys: 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
                 Positive values indicate power leaving the domain.
        
        Example:
            fluxes = sim.compute_poynting_flux_boundaries()
            net_flux = sum(fluxes.values())  # Net power leaving domain
            print(f"Power absorbed by PML: {net_flux:.2e} W")
        """
        Sx, Sy, Sz = self.compute_poynting_vector()
        dA = self.dx**2
        
        fluxes = {
            'x_min': -np.sum(Sx[0, :, :]) * dA,      # Negative for inward flux
            'x_max':  np.sum(Sx[-1, :, :]) * dA,     # Positive for outward flux
            'y_min': -np.sum(Sy[:, 0, :]) * dA,
            'y_max':  np.sum(Sy[:, -1, :]) * dA,
            'z_min': -np.sum(Sz[:, :, 0]) * dA,
            'z_max':  np.sum(Sz[:, :, -1]) * dA,
        }
        
        return fluxes

    def compute_energy_flux_balance(self):
        """
        Compute energy balance: rate of energy change vs boundary fluxes.
        
        For a closed system: dU/dt + ∇·S = -losses
        This method computes the net Poynting flux leaving through boundaries
        to verify energy conservation (useful for PML validation).
        
        Returns:
            dict: Contains 'boundary_flux' (W) and flux per boundary
        
        Note:
            In a well-functioning PML, boundary_flux should be small,
            indicating waves are absorbed rather than reflected.
        """
        fluxes = self.compute_poynting_flux_boundaries()
        net_flux = sum(fluxes.values())
        
        result = {
            'net_boundary_flux': net_flux,
            'boundaries': fluxes,
        }
        
        return result

    # ========== Probe Management Methods ==========
    
    def setup_probes(self):
        """
        Initialise le gestionnaire de sondes.
        
        Returns:
            ProbeManager: Le gestionnaire de sondes créé
        
        Example:
            sim.setup_probes()
            sim.add_probe((nx//2, ny//2, nz//2), 'center')
        """
        from .probes import ProbeManager
        self.probe_manager = ProbeManager()
        logger.info("Probe manager initialized")
        return self.probe_manager
    
    def add_probe(self, position, name=None, record_E=True, record_H=True, record_energy=True):
        """
        Ajoute une sonde à une position donnée.
        
        Parameters:
            position (tuple): (ix, iy, iz) position de la sonde
            name (str, optional): Nom de la sonde
            record_E (bool): Enregistrer les champs électriques
            record_H (bool): Enregistrer les champs magnétiques
            record_energy (bool): Enregistrer l'énergie locale
        
        Returns:
            FieldProbe: La sonde créée
        
        Example:
            probe = sim.add_probe((50, 50, 25), 'center', record_energy=True)
        """
        if self.probe_manager is None:
            self.setup_probes()
        return self.probe_manager.add_probe(position, name, record_E, record_H, record_energy)
    
    def add_line_of_probes(self, start, end, num_probes, name_prefix="line", **kwargs):
        """
        Ajoute une ligne de sondes entre deux points.
        
        Parameters:
            start (tuple): Point de départ (ix, iy, iz)
            end (tuple): Point d'arrivée (ix, iy, iz)
            num_probes (int): Nombre de sondes sur la ligne
            name_prefix (str): Préfixe pour les noms des sondes
            **kwargs: Arguments additionnels (record_E, record_H, record_energy)
        
        Returns:
            List[FieldProbe]: Liste des sondes créées
        
        Example:
            # Ligne de 10 sondes le long de x au centre
            probes = sim.add_line_of_probes((10, ny//2, nz//2), (nx-10, ny//2, nz//2), 10)
        """
        if self.probe_manager is None:
            self.setup_probes()
        return self.probe_manager.add_line_of_probes(start, end, num_probes, name_prefix, **kwargs)
    
    def get_probe_data(self, name=None):
        """
        Récupère les données des sondes.
        
        Parameters:
            name (str, optional): Nom d'une sonde spécifique. 
                                 Si None, retourne toutes les sondes.
        
        Returns:
            dict or Dict[str, dict]: Données de la sonde ou de toutes les sondes
        
        Example:
            data = sim.get_probe_data('center')
            t = data['time']
            Ez = data['Ez']
        """
        if self.probe_manager is None:
            logger.warning("No probe manager initialized")
            return {} if name is None else None
        
        if name is not None:
            probe = self.probe_manager.get_probe(name)
            return probe.get_data() if probe else None
        else:
            return self.probe_manager.get_all_data()
    
    def clear_probe_data(self):
        """Efface l'historique de toutes les sondes."""
        if self.probe_manager is not None:
            self.probe_manager.clear_all()
            logger.info("Cleared all probe data")
    
    def list_probes(self):
        """
        Liste toutes les sondes actives.
        
        Returns:
            List[str]: Noms des sondes
        """
        if self.probe_manager is None:
            return []
        return self.probe_manager.list_probes()

if __name__ == '__main__':
    print('Yee3D module loaded. Use from tests or scripts.')
