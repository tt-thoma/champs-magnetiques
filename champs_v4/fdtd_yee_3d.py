#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

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
from typing import Optional

import importlib.util

NUMBA = importlib.util.find_spec("numba") is not None
if NUMBA:
    from numba import njit, prange


logger = logging.getLogger(__name__)

# Physical constants
epsilon0 = 8.8541878128e-12
mu0 = 4 * math.pi * 1e-7
c0 = 1.0 / math.sqrt(epsilon0 * mu0)


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
            *, use_numba: bool = False, pml_width: int = 10, pml_sigma_max: float = 1.0
    ):
        """
        Initialize a 3D Yee grid.
        nx,ny,nz : number of cells in each direction (integer)
        dx : spatial step (assumed equal in all directions)
        dt : time step (should satisfy CFL)
        """
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.dx = float(dx)
        self.dt = float(dt)

        # fields: using Yee stagger: E components centered on edges,
        # H components centered on faces. We'll store arrays sized so
        # that finite difference indexing is straightforward.
        self.Ex = np.zeros((nx, ny + 1, nz + 1), dtype=np.float64)
        self.Ey = np.zeros((nx + 1, ny, nz + 1), dtype=np.float64)
        self.Ez = np.zeros((nx + 1, ny + 1, nz), dtype=np.float64)

        self.Hx = np.zeros((nx + 1, ny, nz), dtype=np.float64)
        self.Hy = np.zeros((nx, ny + 1, nz), dtype=np.float64)
        self.Hz = np.zeros((nx, ny, nz + 1), dtype=np.float64)

        # material maps (cell-centered)
        self.epsilon_r = np.ones((nx, ny, nz), dtype=np.float64)
        self.sigma = np.zeros((nx, ny, nz), dtype=np.float64)

        # macroscopic current density on staggered grid corresponding to E
        self.Jx = np.zeros_like(self.Ex)
        self.Jy = np.zeros_like(self.Ey)
        self.Jz = np.zeros_like(self.Ez)

        # Derived coefficients for update (will be computed)
        self._compute_material_coeffs()

        if use_numba and not NUMBA:
            logger.warning("use_numba was enabled while the numba package has not been installed.")

        # optional numba
        self.use_numba = use_numba and NUMBA
        if self.use_numba:
            # prepare numba compiled kernels if available
            try:
                self._prepare_numba_kernels()
            except Exception as err:
                logger.error("Could not set up numba", exc_info=err)
                self.use_numba = False

        # PML (simple exponential damping mask) parameters
        self.pml_width = int(pml_width)
        self.pml_sigma_max = float(pml_sigma_max)
        self._init_pml()

    def _compute_material_coeffs(self):
        # coefficients at E locations: need epsilon and sigma at E positions.
        # For simplicity, interpolate cell-centered epsilon/sigma to edge locations
        # via averaging neighbor cells.
        nx, ny, nz = self.nx, self.ny, self.nz
        # create arrays for eps and sigma at E-grid shapes
        self.eps_Ex = np.zeros_like(self.Ex)
        self.sig_Ex = np.zeros_like(self.Ex)

        for i in range(nx):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    # neighbors cell indices (clamp)
                    il = min(max(i, 0), nx - 1)
                    jl = min(max(j - 1, 0), ny - 1)
                    kl = min(max(k - 1, 0), nz - 1)
                    # average over up to 4 surrounding cells
                    vals_eps = []
                    vals_sig = []
                    for ii in [il, min(il + 1, nx - 1)]:
                        for jj in [jl, min(jl + 1, ny - 1)]:
                            for kk in [kl, min(kl + 1, nz - 1)]:
                                vals_eps.append(self.epsilon_r[ii, jj, kk])
                                vals_sig.append(self.sigma[ii, jj, kk])
                    self.eps_Ex[i, j, k] = np.mean(vals_eps)
                    self.sig_Ex[i, j, k] = np.mean(vals_sig)

        self.eps_Ey = np.zeros_like(self.Ey)
        self.sig_Ey = np.zeros_like(self.Ey)
        for i in range(nx + 1):
            for j in range(ny):
                for k in range(nz + 1):
                    il = min(max(i - 1, 0), nx - 1)
                    jl = min(max(j, 0), ny - 1)
                    kl = min(max(k - 1, 0), nz - 1)
                    vals_eps = []
                    vals_sig = []
                    for ii in [il, min(il + 1, nx - 1)]:
                        for jj in [jl, min(jl + 1, ny - 1)]:
                            for kk in [kl, min(kl + 1, nz - 1)]:
                                vals_eps.append(self.epsilon_r[ii, jj, kk])
                                vals_sig.append(self.sigma[ii, jj, kk])
                    self.eps_Ey[i, j, k] = np.mean(vals_eps)
                    self.sig_Ey[i, j, k] = np.mean(vals_sig)

        self.eps_Ez = np.zeros_like(self.Ez)
        self.sig_Ez = np.zeros_like(self.Ez)
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz):
                    il = min(max(i - 1, 0), nx - 1)
                    jl = min(max(j - 1, 0), ny - 1)
                    kl = min(max(k, 0), nz - 1)
                    vals_eps = []
                    vals_sig = []
                    for ii in [il, min(il + 1, nx - 1)]:
                        for jj in [jl, min(jl + 1, ny - 1)]:
                            for kk in [kl, min(kl + 1, nz - 1)]:
                                vals_eps.append(self.epsilon_r[ii, jj, kk])
                                vals_sig.append(self.sigma[ii, jj, kk])
                    self.eps_Ez[i, j, k] = np.mean(vals_eps)
                    self.sig_Ez[i, j, k] = np.mean(vals_sig)

        # Precompute update multipliers for E fields: accounting for conductivity
        self.cex = (1.0 / (epsilon0 * self.eps_Ex)) * self.dt
        self.cey = (1.0 / (epsilon0 * self.eps_Ey)) * self.dt
        self.cez = (1.0 / (epsilon0 * self.eps_Ez)) * self.dt

    def set_materials(self, epsilon_r: np.ndarray, sigma: Optional[np.ndarray] = None):
        """
        Set material properties for the grid.

        Parameters:
            epsilon_r (np.ndarray): Relative permittivity map, shape (nx, ny, nz).
            sigma (np.ndarray, optional): Conductivity map, shape (nx, ny, nz). Defaults to zeros.

        Raises:
            AssertionError: If array shapes do not match (nx, ny, nz).
        """
        assert epsilon_r.shape == (self.nx, self.ny, self.nz)

        self.epsilon_r = epsilon_r.astype(np.float64)

        if sigma is not None:
            assert sigma.shape == (self.nx, self.ny, self.nz)
            self.sigma = sigma.astype(np.float64)

        self._compute_material_coeffs()

    def add_coil(self, center, radius_cells, axis='z', turns=1, current=1.0):
        """
        Approximate a coil (solenoid) by adding macroscopic current density J
        along axis direction distributed over the loop cells.
        center : (ix,iy,iz) cell indices for coil center
        radius_cells : radius in cells
        axis : 'x','y' or 'z'
        turns : number of turns (multiplicative factor)
        current : current amplitude (A)
        """
        cx, cy, cz = center
        ix0, iy0, iz0 = int(cx), int(cy), int(cz)

        # use Jz for axis == 'z', etc.
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    # distance in perpendicular plane
                    if axis == 'z':
                        dx = i - ix0
                        dy = j - iy0

                        if dx * dx + dy * dy <= radius_cells * radius_cells:
                            # set Jz at Ez staggered positions nearby
                            iz_idx = k
                            if (
                                    0 <= i < self.Ez.shape[0]
                                    and 0 <= j < self.Ez.shape[1]
                                    and 0 <= iz_idx < self.Ez.shape[2]
                            ):
                                self.Jz[i, j, iz_idx] += current * turns

                    # other axes can be implemented similarly

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
        # Hx(i,j,k) += -(dt/(mu0)) * (dEz/dy - dEy/dz)
        dt = self.dt
        dx = self.dx
        coef = dt / mu0
        # compute derivatives via differences
        # Hx update
        # derivative dEz/dy
        dEz_dy = (self.Ez[:-1, 1:, :] - self.Ez[:-1, :-1, :]) / dx
        dEy_dz = (self.Ey[:-1, :, 1:] - self.Ey[:-1, :, :-1]) / dx
        # apply CPML auxiliary psi (if initialized)
        if hasattr(self, 'psi_ez_dy'):
            self.psi_ez_dy = self.b_ez_dy * self.psi_ez_dy + self.c_ez_dy * dEz_dy
            dEz_eff = dEz_dy + self.psi_ez_dy
        else:
            dEz_eff = dEz_dy
        if hasattr(self, 'psi_ey_dz'):
            self.psi_ey_dz = self.b_ey_dz * self.psi_ey_dz + self.c_ey_dz * dEy_dz
            dEy_eff = dEy_dz + self.psi_ey_dz
        else:
            dEy_eff = dEy_dz
        self.Hx[:-1, :, :] += -coef * (dEz_eff - dEy_eff)
        # Hy update
        dEx_dz = (self.Ex[:, :-1, 1:] - self.Ex[:, :-1, :-1]) / dx
        dEz_dx = (self.Ez[1:, :-1, :] - self.Ez[:-1, :-1, :]) / dx
        if hasattr(self, 'psi_ex_dz'):
            self.psi_ex_dz = self.b_ex_dz * self.psi_ex_dz + self.c_ex_dz * dEx_dz
            dEx_eff = dEx_dz + self.psi_ex_dz
        else:
            dEx_eff = dEx_dz
        if hasattr(self, 'psi_ez_dx'):
            self.psi_ez_dx = self.b_ez_dx * self.psi_ez_dx + self.c_ez_dx * dEz_dx
            dEz_eff2 = dEz_dx + self.psi_ez_dx
        else:
            dEz_eff2 = dEz_dx
        self.Hy[:, :-1, :] += -coef * (dEx_eff - dEz_eff2)
        # Hz update
        dEy_dx = (self.Ey[1:, :, :-1] - self.Ey[:-1, :, :-1]) / dx
        dEx_dy = (self.Ex[:, 1:, :-1] - self.Ex[:, :-1, :-1]) / dx
        if hasattr(self, 'psi_ey_dx'):
            self.psi_ey_dx = self.b_ey_dx * self.psi_ey_dx + self.c_ey_dx * dEy_dx
            dEy_eff2 = dEy_dx + self.psi_ey_dx
        else:
            dEy_eff2 = dEy_dx
        if hasattr(self, 'psi_ex_dy'):
            self.psi_ex_dy = self.b_ex_dy * self.psi_ex_dy + self.c_ex_dy * dEx_dy
            dEx_eff2 = dEx_dy + self.psi_ex_dy
        else:
            dEx_eff2 = dEx_dy
        self.Hz[:, :, :-1] += -coef * (dEy_eff2 - dEx_eff2)

    def update_E(self):
        # E updates include conductivity and source J (at E locations)
        dt = self.dt
        dx = self.dx
        coefH = dt / (epsilon0)
        # Ex update on indices Ex[0:nx, 1:ny, 1:nz]
        nx, ny, nz = self.nx, self.ny, self.nz
        if nx > 0 and ny > 1 and nz > 1:
            curlHx = (self.Hz[0:nx, 1:ny, 1:nz] - self.Hz[0:nx, 0:ny-1, 1:nz]) / dx - (
                self.Hy[0:nx, 1:ny, 1:nz] - self.Hy[0:nx, 1:ny, 0:nz-1]
            ) / dx
            # Stable update accounting for conductivity (semi-implicit)
            eps_local = epsilon0 * self.eps_Ex[0:nx, 1:ny, 1:nz] + 1e-30
            sigma_local = self.sig_Ex[0:nx, 1:ny, 1:nz]
            alpha = (sigma_local * dt) / (2.0 * eps_local)
            denom = 1.0 + alpha
            numer_factor = 1.0 - alpha
            # incorporate CPML psi for derivatives of H entering Ex update
            # dHz/dy and dHy/dz terms used in curlHx have shapes matching curlHx
            dHz_dy = (self.Hz[0:nx, 1:ny, 1:nz] - self.Hz[0:nx, 0:ny-1, 1:nz]) / dx
            dHy_dz = (self.Hy[0:nx, 1:ny, 1:nz] - self.Hy[0:nx, 1:ny, 0:nz-1]) / dx
            if hasattr(self, 'psi_hz_dy'):
                # psi_hz_dy shape should match (nx,ny,nz)
                self.psi_hz_dy = self.b_hz_dy * self.psi_hz_dy + self.c_hz_dy * dHz_dy
                dHz_eff = dHz_dy + self.psi_hz_dy
            else:
                dHz_eff = dHz_dy
            if hasattr(self, 'psi_hy_dz'):
                self.psi_hy_dz = self.b_hy_dz * self.psi_hy_dz + self.c_hy_dz * dHy_dz
                dHy_eff = dHy_dz + self.psi_hy_dz
            else:
                dHy_eff = dHy_dz
            curlHx_eff = dHz_eff - dHy_eff
            rhs = (curlHx_eff - self.Jx[0:nx, 1:ny, 1:nz]) * (dt / eps_local)
            self.Ex[0:nx, 1:ny, 1:nz] = (numer_factor * self.Ex[0:nx, 1:ny, 1:nz] + rhs) / denom

        # Ey update on indices Ey[1:nx, 0:ny, 1:nz]
        if nx > 1 and ny > 0 and nz > 1:
            curlHy = (self.Hx[1:nx, 0:ny, 1:nz] - self.Hx[1:nx, 0:ny, 0:nz-1]) / dx - (
                self.Hz[1:nx, 0:ny, 1:nz] - self.Hz[0:nx-1, 0:ny, 1:nz]
            ) / dx
            eps_local = epsilon0 * self.eps_Ey[1:nx, 0:ny, 1:nz] + 1e-30
            sigma_local = self.sig_Ey[1:nx, 0:ny, 1:nz]
            alpha = (sigma_local * dt) / (2.0 * eps_local)
            denom = 1.0 + alpha
            numer_factor = 1.0 - alpha
            # include psi terms for H derivatives entering Ey
            dHx_dz = (self.Hx[1:nx, 0:ny, 1:nz] - self.Hx[1:nx, 0:ny, 0:nz-1]) / dx
            dHz_dx = (self.Hz[1:nx, 0:ny, 1:nz] - self.Hz[0:nx-1, 0:ny, 1:nz]) / dx
            if hasattr(self, 'psi_hx_dz'):
                self.psi_hx_dz = self.b_hx_dz * self.psi_hx_dz + self.c_hx_dz * dHx_dz
                dHx_eff = dHx_dz + self.psi_hx_dz
            else:
                dHx_eff = dHx_dz
            if hasattr(self, 'psi_hz_dx'):
                self.psi_hz_dx = self.b_hz_dx * self.psi_hz_dx + self.c_hz_dx * dHz_dx
                dHz_eff2 = dHz_dx + self.psi_hz_dx
            else:
                dHz_eff2 = dHz_dx
            curlHy_eff = dHx_eff - dHz_eff2
            rhs = (curlHy_eff - self.Jy[1:nx, 0:ny, 1:nz]) * (dt / eps_local)
            self.Ey[1:nx, 0:ny, 1:nz] = (numer_factor * self.Ey[1:nx, 0:ny, 1:nz] + rhs) / denom

        # Ez update on indices Ez[1:nx, 1:ny, 0:nz]
        if nx > 1 and ny > 1 and nz > 0:
            curlHz = (self.Hy[1:nx, 1:ny, 0:nz] - self.Hy[0:nx-1, 1:ny, 0:nz]) / dx - (
                self.Hx[1:nx, 1:ny, 0:nz] - self.Hx[1:nx, 0:ny-1, 0:nz]
            ) / dx
            eps_local = epsilon0 * self.eps_Ez[1:nx, 1:ny, 0:nz] + 1e-30
            sigma_local = self.sig_Ez[1:nx, 1:ny, 0:nz]
            alpha = (sigma_local * dt) / (2.0 * eps_local)
            denom = 1.0 + alpha
            numer_factor = 1.0 - alpha
            # include H-derivative psi terms for Ez update
            dHy_dx = (self.Hy[1:nx, 1:ny, 0:nz] - self.Hy[0:nx-1, 1:ny, 0:nz]) / dx
            dHx_dy = (self.Hx[1:nx, 1:ny, 0:nz] - self.Hx[1:nx, 0:ny-1, 0:nz]) / dx
            if hasattr(self, 'psi_hy_dx'):
                self.psi_hy_dx = self.b_hy_dx * self.psi_hy_dx + self.c_hy_dx * dHy_dx
                dHy_eff3 = dHy_dx + self.psi_hy_dx
            else:
                dHy_eff3 = dHy_dx
            if hasattr(self, 'psi_hx_dy'):
                self.psi_hx_dy = self.b_hx_dy * self.psi_hx_dy + self.c_hx_dy * dHx_dy
                dHx_eff3 = dHx_dy + self.psi_hx_dy
            else:
                dHx_eff3 = dHx_dy
            curlHz_eff = dHy_eff3 - dHx_eff3
            rhs = (curlHz_eff - self.Jz[1:nx, 1:ny, 0:nz]) * (dt / eps_local)
            self.Ez[1:nx, 1:ny, 0:nz] = (numer_factor * self.Ez[1:nx, 1:ny, 0:nz] + rhs) / denom

    def _prepare_numba_kernels(self):
        # Placeholder: for larger grids we would implement numba-compiled loops
        # to update H and E. For now we leave it as potential extension.
        pass

    def _init_pml(self):
        """
        Initialize simple exponential damping masks for E and H fields.
        This is a lightweight PML substitute: it applies an exponential
        damping mask near boundaries to absorb outgoing waves. Not a full
        CPML implementation, but effective for reducing reflections.
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        w = max(0, int(self.pml_width))
        sigma_max = float(self.pml_sigma_max)

        def axis_sigma(L):
            s = np.zeros(L, dtype=float)
            if w <= 0:
                return s
            # limit local width to at most half the axis length
            w_local = min(w, L // 2)
            for idx in range(w_local):
                frac = (w_local - idx) / float(max(1, w_local))
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
        sigma_ez_dy = 0.5 * (sigma_total_Ez[:-1, 1:, :] + sigma_total_Ez[:-1, :-1, :])
        sigma_ey_dz = 0.5 * (sigma_total_Ey[:-1, :, 1:] + sigma_total_Ey[:-1, :, :-1])
        sigma_ex_dz = 0.5 * (sigma_total_Ex[:, :-1, 1:] + sigma_total_Ex[:, :-1, :-1])
        sigma_ez_dx = 0.5 * (sigma_total_Ez[1:, :-1, :] + sigma_total_Ez[:-1, :-1, :])
        sigma_ey_dx = 0.5 * (sigma_total_Ey[1:, :, :-1] + sigma_total_Ey[:-1, :, :-1])
        sigma_ex_dy = 0.5 * (sigma_total_Ex[:, 1:, :-1] + sigma_total_Ex[:, :-1, :-1])

        # For derivatives of H, use H sigma_total analogs (sigma_total_Hx etc.)
        # Use the exact same slices as the derivatives in update_E to build sigma
        sigma_hz_dy = 0.5 * (sigma_total_Hz[0:self.nx, 1:self.ny, 1:self.nz] + sigma_total_Hz[0:self.nx, 0:self.ny-1, 1:self.nz])
        sigma_hy_dz = 0.5 * (sigma_total_Hy[0:self.nx, 1:self.ny, 1:self.nz] + sigma_total_Hy[0:self.nx, 1:self.ny, 0:self.nz-1])
        sigma_hx_dz = 0.5 * (sigma_total_Hx[1:self.nx, 0:self.ny, 1:self.nz] + sigma_total_Hx[1:self.nx, 0:self.ny, 0:self.nz-1])
        sigma_hz_dx = 0.5 * (sigma_total_Hz[1:self.nx, 0:self.ny, 1:self.nz] + sigma_total_Hz[0:self.nx-1, 0:self.ny, 1:self.nz])
        sigma_hy_dx = 0.5 * (sigma_total_Hy[1:self.nx, 1:self.ny, 0:self.nz] + sigma_total_Hy[0:self.nx-1, 1:self.ny, 0:self.nz])
        sigma_hx_dy = 0.5 * (sigma_total_Hx[1:self.nx, 1:self.ny, 0:self.nz] + sigma_total_Hx[1:self.nx, 0:self.ny-1, 0:self.nz])

        # Ensure shapes equal to (nx,ny,nz)
        # Recursion coefficients b = exp(-sigma*dt/param), c = (1-b)/sigma (approx)
        tiny = 1e-30
        b_scale_e = dt / epsilon0
        b_scale_h = dt / mu0

        def make_bc(sigma_arr, scale):
            b = np.exp(-sigma_arr * scale)
            # c: avoid divide by zero: if sigma small, c ~ scale (i.e., dt/eps)
            with np.errstate(divide='ignore', invalid='ignore'):
                c = np.where(sigma_arr > 0, (1.0 - b) / (sigma_arr + tiny), scale)
            return b, c

        self.b_ez_dy, self.c_ez_dy = make_bc(sigma_ez_dy, b_scale_e)
        self.b_ey_dz, self.c_ey_dz = make_bc(sigma_ey_dz, b_scale_e)
        self.b_ex_dz, self.c_ex_dz = make_bc(sigma_ex_dz, b_scale_e)
        self.b_ez_dx, self.c_ez_dx = make_bc(sigma_ez_dx, b_scale_e)
        self.b_ey_dx, self.c_ey_dx = make_bc(sigma_ey_dx, b_scale_e)
        self.b_ex_dy, self.c_ex_dy = make_bc(sigma_ex_dy, b_scale_e)

        self.b_hz_dy, self.c_hz_dy = make_bc(sigma_hz_dy, b_scale_h)
        self.b_hy_dz, self.c_hy_dz = make_bc(sigma_hy_dz, b_scale_h)
        self.b_hx_dz, self.c_hx_dz = make_bc(sigma_hx_dz, b_scale_h)
        self.b_hz_dx, self.c_hz_dx = make_bc(sigma_hz_dx, b_scale_h)
        self.b_hy_dx, self.c_hy_dx = make_bc(sigma_hy_dx, b_scale_h)
        self.b_hx_dy, self.c_hx_dy = make_bc(sigma_hx_dy, b_scale_h)

        # Initialize psi arrays with shapes matching sigma derivative arrays
        self.psi_ez_dy = np.zeros_like(sigma_ez_dy)
        self.psi_ey_dz = np.zeros_like(sigma_ey_dz)
        self.psi_ex_dz = np.zeros_like(sigma_ex_dz)
        self.psi_ez_dx = np.zeros_like(sigma_ez_dx)
        self.psi_ey_dx = np.zeros_like(sigma_ey_dx)
        self.psi_ex_dy = np.zeros_like(sigma_ex_dy)

        self.psi_hz_dy = np.zeros_like(sigma_hz_dy)
        self.psi_hy_dz = np.zeros_like(sigma_hy_dz)
        self.psi_hx_dz = np.zeros_like(sigma_hx_dz)
        self.psi_hz_dx = np.zeros_like(sigma_hz_dx)
        self.psi_hy_dx = np.zeros_like(sigma_hy_dx)
        self.psi_hx_dy = np.zeros_like(sigma_hx_dy)


if __name__ == '__main__':
    print('Yee3D module loaded. Use from tests or scripts.')
