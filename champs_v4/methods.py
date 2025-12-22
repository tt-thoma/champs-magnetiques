#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# SenfinaLight, VirinasCode, tt_thoma

import numpy as np
from .constants import epsilon0, mu0
from .config import int_t, float_t, range_f, ndarray_t, njit

@njit(cache=True)
def compute_material_coefficients(
        nx: int_t,
        ny: int_t,
        nz: int_t,

        Ex: ndarray_t,
        Ey: ndarray_t,
        Ez: ndarray_t,

        epsilon_r: ndarray_t,
        sigma: ndarray_t,

        dt: float_t
):
    """
    Computes coefficients at E locations.
    Need epsilon and sigma at E positions.

    Note: For simplicity, interpolate cell-centered epsilon/sigma to edge locations
    via averaging neighbor cells.
    """

    # create arrays for eps and sigma at E-grid shapes
    epsilon_Ex = np.zeros_like(Ex)
    sigma_Ex = np.zeros_like(Ex)

    for x in range_f(nx):
        for y in range_f(ny + 1):
            for z in range_f(nz + 1):
                # neighbors cell indices (clamp)
                x_left = min(max(x, 0), nx - 1)
                y_top = min(max(y - 1, 0), ny - 1)
                z_back = min(max(z - 1, 0), nz - 1)

                # average over up to 4 surrounding cells
                vals_epsilon = np.zeros((8,), dtype=float_t)
                vals_sigma = np.zeros((8,), dtype=float_t)

                idx = 0
                for ii in [x_left, min(x_left + 1, nx - 1)]:
                    for jj in [y_top, min(y_top + 1, ny - 1)]:
                        for kk in [z_back, min(z_back + 1, nz - 1)]:
                            vals_epsilon[idx] = epsilon_r[ii, jj, kk]
                            vals_sigma[idx] = sigma[ii, jj, kk]
                            idx += 1

                epsilon_Ex[x, y, z] = np.mean(vals_epsilon)
                sigma_Ex[x, y, z] = np.mean(vals_sigma)

    epsilon_Ey = np.zeros_like(Ey)
    sigma_Ey = np.zeros_like(Ey)

    for x in range_f(nx + 1):
        for y in range_f(ny):
            for z in range_f(nz + 1):
                x_left = min(max(x - 1, 0), nx - 1)
                y_top = min(max(y, 0), ny - 1)
                z_back = min(max(z - 1, 0), nz - 1)

                vals_epsilon = np.zeros((8,), dtype=float_t)
                vals_sigma = np.zeros((8,), dtype=float_t)

                idx = 0
                for ii in [x_left, min(x_left + 1, nx - 1)]:
                    for jj in [y_top, min(y_top + 1, ny - 1)]:
                        for kk in [z_back, min(z_back + 1, nz - 1)]:
                            vals_epsilon[idx] = epsilon_r[ii, jj, kk]
                            vals_sigma[idx] = sigma[ii, jj, kk]
                            idx += 1

                epsilon_Ey[x, y, z] = np.mean(vals_epsilon)
                sigma_Ey[x, y, z] = np.mean(vals_sigma)

    epsilon_Ez = np.zeros_like(Ez)
    sigma_Ez = np.zeros_like(Ez)

    for x in range_f(nx + 1):
        for y in range_f(ny + 1):
            for z in range_f(nz):
                x_left = min(max(x - 1, 0), nx - 1)
                y_top = min(max(y - 1, 0), ny - 1)
                z_back = min(max(z, 0), nz - 1)

                vals_epsilon = np.zeros((8,), dtype=float_t)
                vals_sigma = np.zeros((8,), dtype=float_t)

                idx = 0
                for ii in [x_left, min(x_left + 1, nx - 1)]:
                    for jj in [y_top, min(y_top + 1, ny - 1)]:
                        for kk in [z_back, min(z_back + 1, nz - 1)]:
                            vals_epsilon[idx] = epsilon_r[ii, jj, kk]
                            vals_sigma[idx] = sigma[ii, jj, kk]
                            idx += 1

                epsilon_Ez[x, y, z] = np.mean(vals_epsilon)
                sigma_Ez[x, y, z] = np.mean(vals_sigma)

    # Precompute update multipliers for E fields: accounting for conductivity
    cex = (1.0 / (epsilon0 * epsilon_Ex)) * dt
    cey = (1.0 / (epsilon0 * epsilon_Ey)) * dt
    cez = (1.0 / (epsilon0 * epsilon_Ez)) * dt

    return (
        epsilon_Ex, epsilon_Ey, epsilon_Ez,
        sigma_Ex, sigma_Ey, sigma_Ez,
        cex, cey, cez
    )

# @njit
def set_materials(
        nx: int_t,
        ny: int_t,
        nz: int_t,

        epsilon_r: ndarray_t,
        sigma: ndarray_t = None
):
    epsilon_r = epsilon_r.astype(float_t)

    if sigma is not None:
        sigma = sigma.astype(float_t)

    return epsilon_r, sigma

# @njit
def add_coil(
        nx: int_t,
        ny: int_t,
        nz: int_t,

        Ez: ndarray_t,
        Jz: ndarray_t,

        center: tuple[int_t, int_t, int_t],
        radius_cells,

        *,
        axis: str = 'z',
        turns: int_t = 1,
        current: float_t = 1.0
):
    cx, cy, cz = center
    ix0, iy0, iz0 = int_t(cx), int_t(cy), int_t(cz)

    # use Jz for axis == 'z', etc.
    for i in range_f(nx):
        for j in range_f(ny):
            for k in range_f(nz):
                # distance in perpendicular plane
                if axis == 'z':
                    dx = i - ix0
                    dy = j - iy0

                    if dx * dx + dy * dy <= radius_cells * radius_cells:
                        # set Jz at Ez staggered positions nearby
                        iz_idx = k
                        if (
                                0 <= i < Ez.shape[0]
                                and 0 <= j < Ez.shape[1]
                                and 0 <= iz_idx < Ez.shape[2]
                        ):
                            Jz[i, j, iz_idx] += current * turns

    return Jz

@njit(cache=True)
def step(
        nx: int_t,
        ny: int_t,
        nz: int_t,

        Ex: ndarray_t,
        Ey: ndarray_t,
        Ez: ndarray_t,

        Hx: ndarray_t,
        Hy: ndarray_t,
        Hz: ndarray_t,

        Jx: ndarray_t,
        Jy: ndarray_t,
        Jz: ndarray_t,

        dampE: ndarray_t,
        dampE_Ex: ndarray_t,
        dampE_Ey: ndarray_t,
        dampE_Ez: ndarray_t,

        epsilon_Ex: ndarray_t,
        epsilon_Ey: ndarray_t,
        epsilon_Ez: ndarray_t,

        sigma_Ex: ndarray_t,
        sigma_Ey: ndarray_t,
        sigma_Ez: ndarray_t,

        psi_ex_dy: ndarray_t,
        psi_ex_dz: ndarray_t,

        psi_ey_dx: ndarray_t,
        psi_ey_dz: ndarray_t,

        psi_ez_dx: ndarray_t,
        psi_ez_dy: ndarray_t,

        b_ex_dy: ndarray_t,
        c_ex_dy: ndarray_t,
        kappa_ex_dy: ndarray_t,

        b_ex_dz: ndarray_t,
        c_ex_dz: ndarray_t,
        kappa_ex_dz: ndarray_t,

        b_ey_dx: ndarray_t,
        c_ey_dx: ndarray_t,
        kappa_ey_dx: ndarray_t,

        b_ey_dz: ndarray_t,
        c_ey_dz: ndarray_t,
        kappa_ey_dz: ndarray_t,

        b_ez_dx: ndarray_t,
        c_ez_dx: ndarray_t,
        kappa_ez_dx: ndarray_t,

        b_ez_dy: ndarray_t,
        c_ez_dy: ndarray_t,
        kappa_ez_dy: ndarray_t,

        psi_hx_dy: ndarray_t,
        psi_hx_dz: ndarray_t,

        psi_hy_dx: ndarray_t,
        psi_hy_dz: ndarray_t,

        psi_hz_dx: ndarray_t,
        psi_hz_dy: ndarray_t,

        b_hx_dy: ndarray_t,
        c_hx_dy: ndarray_t,
        kappa_hx_dy: ndarray_t,

        b_hx_dz: ndarray_t,
        c_hx_dz: ndarray_t,
        kappa_hx_dz: ndarray_t,

        b_hy_dx: ndarray_t,
        c_hy_dx: ndarray_t,
        kappa_hy_dx: ndarray_t,

        b_hy_dz: ndarray_t,
        c_hy_dz: ndarray_t,
        kappa_hy_dz: ndarray_t,

        b_hz_dx: ndarray_t,
        c_hz_dx: ndarray_t,
        kappa_hz_dx: ndarray_t,

        b_hz_dy: ndarray_t,
        c_hz_dy: ndarray_t,
        kappa_hz_dy: ndarray_t,

        dampH_Hx: ndarray_t,
        dampH_Hy: ndarray_t,
        dampH_Hz: ndarray_t,

        dt: float_t, dx: float_t
):
    #---- BEGIN UPDATE_H ----
    coef = dt / mu0

    # compute derivatives via differences
    # Hx update
    # derivative dEz/dy
    dEz_dy = (Ez[:-1, 1:, :] - Ez[:-1, :-1, :]) / dx
    dEy_dz = (Ey[:-1, :, 1:] - Ey[:-1, :, :-1]) / dx

    # apply CPML auxiliary psi (if initialized)
    if psi_ez_dy is not None:
        psi_ez_dy = b_ez_dy * psi_ez_dy + c_ez_dy * dEz_dy
        dEz_eff = dEz_dy / kappa_ez_dy + psi_ez_dy
    else:
        dEz_eff = dEz_dy

    if psi_ey_dz is not None:
        psi_ey_dz = b_ey_dz * psi_ey_dz + c_ey_dz * dEy_dz
        dEy_eff = dEy_dz / kappa_ey_dz + psi_ey_dz
    else:
        dEy_eff = dEy_dz

    Hx[:-1, :, :] += -coef * (dEz_eff - dEy_eff)
    # Hy update
    dEx_dz = (Ex[:, :-1, 1:] - Ex[:, :-1, :-1]) / dx
    dEz_dx = (Ez[1:, :-1, :] - Ez[:-1, :-1, :]) / dx

    if psi_ex_dz is not None:
        psi_ex_dz = b_ex_dz * psi_ex_dz + c_ex_dz * dEx_dz
        dEx_eff = dEx_dz / kappa_ex_dz + psi_ex_dz
    else:
        dEx_eff = dEx_dz

    if psi_ez_dx is not None:
        psi_ez_dx = b_ez_dx * psi_ez_dx + c_ez_dx * dEz_dx
        dEz_eff2 = dEz_dx / kappa_ez_dx + psi_ez_dx
    else:
        dEz_eff2 = dEz_dx

    Hy[:, :-1, :] += -coef * (dEx_eff - dEz_eff2)
    # Hz update
    dEy_dx = (Ey[1:, :, :-1] - Ey[:-1, :, :-1]) / dx
    dEx_dy = (Ex[:, 1:, :-1] - Ex[:, :-1, :-1]) / dx

    if psi_ey_dx is not None:
        psi_ey_dx = b_ey_dx * psi_ey_dx + c_ey_dx * dEy_dx
        dEy_eff2 = dEy_dx / kappa_ey_dx + psi_ey_dx
    else:
        dEy_eff2 = dEy_dx

    if psi_ex_dy is not None:
        psi_ex_dy = b_ex_dy * psi_ex_dy + c_ex_dy * dEx_dy
        dEx_eff2 = dEx_dy / kappa_ex_dy + psi_ex_dy
    else:
        dEx_eff2 = dEx_dy

    Hz[:, :, :-1] += -coef * (dEy_eff2 - dEx_eff2)

    #---- END UPDATE_H ----

    # apply simple H damping in PML
    if dampH_Hx is not None:
        Hx *= dampH_Hx
    if dampH_Hy is not None:
        Hy *= dampH_Hy
    if dampH_Hz is not None:
        Hz *= dampH_Hz

    #---- BEGIN UPDATE_E ----
    # E updates include conductivity and source J (at E locations)
    # Ex update on indices Ex[0:nx, 1:ny, 1:nz)

    if nx > 0 and ny > 1 and nz > 1:
        # Stable update accounting for conductivity (semi-implicit)
        eps_local = epsilon0 * epsilon_Ex[0:nx, 1:ny, 1:nz] + 1e-30
        sigma_local = sigma_Ex[0:nx, 1:ny, 1:nz]
        alpha = (sigma_local * dt) / (2.0 * eps_local)
        denom = 1.0 + alpha
        numer_factor = 1.0 - alpha

        # incorporate CPML psi for derivatives of H entering Ex update
        # dHz/dy and dHy/dz terms used in curlHx have shapes matching curlHx
        dHz_dy = (Hz[0:nx, 1:ny, 1:nz] - Hz[0:nx, 0:ny-1, 1:nz]) / dx
        dHy_dz = (Hy[0:nx, 1:ny, 1:nz] - Hy[0:nx, 1:ny, 0:nz-1]) / dx

        if psi_hz_dy is not None:
            # psi_hz_dy shape should match (nx,ny,nz)
            psi_hz_dy = b_hz_dy * psi_hz_dy + c_hz_dy * dHz_dy
            dHz_eff = dHz_dy / kappa_hz_dy + psi_hz_dy
        else:
            dHz_eff = dHz_dy

        if psi_hy_dz is not None:
            psi_hy_dz = b_hy_dz * psi_hy_dz + c_hy_dz * dHy_dz
            dHy_eff = dHy_dz / kappa_hy_dz + psi_hy_dz
        else:
            dHy_eff = dHy_dz

        curlHx_eff = dHz_eff - dHy_eff
        rhs = (curlHx_eff - Jx[0:nx, 1:ny, 1:nz]) * (dt / eps_local)
        Ex[0:nx, 1:ny, 1:nz] = (numer_factor * Ex[0:nx, 1:ny, 1:nz] + rhs) / denom

    # Ey update on indices Ey[1:nx, 0:ny, 1:nz]
    if nx > 1 and ny > 0 and nz > 1:
        eps_local = epsilon0 * epsilon_Ey[1:nx, 0:ny, 1:nz] + 1e-30
        sigma_local = sigma_Ey[1:nx, 0:ny, 1:nz]
        alpha = (sigma_local * dt) / (2.0 * eps_local)
        denom = 1.0 + alpha
        numer_factor = 1.0 - alpha

        # include psi terms for H derivatives entering Ey
        dHx_dz = (Hx[1:nx, 0:ny, 1:nz] - Hx[1:nx, 0:ny, 0:nz-1]) / dx
        dHz_dx = (Hz[1:nx, 0:ny, 1:nz] - Hz[0:nx-1, 0:ny, 1:nz]) / dx

        if psi_hx_dz is not None:
            psi_hx_dz = b_hx_dz * psi_hx_dz + c_hx_dz * dHx_dz
            dHx_eff = dHx_dz / kappa_hx_dz + psi_hx_dz
        else:
            dHx_eff = dHx_dz
        if psi_hz_dx is not None:
            psi_hz_dx = b_hz_dx * psi_hz_dx + c_hz_dx * dHz_dx
            dHz_eff2 = dHz_dx / kappa_hz_dx + psi_hz_dx
        else:
            dHz_eff2 = dHz_dx
        curlHy_eff = dHx_eff - dHz_eff2
        rhs = (curlHy_eff - Jy[1:nx, 0:ny, 1:nz]) * (dt / eps_local)
        Ey[1:nx, 0:ny, 1:nz] = (numer_factor * Ey[1:nx, 0:ny, 1:nz] + rhs) / denom

    # Ez update on indices Ez[1:nx, 1:ny, 0:nz]
    if nx > 1 and ny > 1 and nz > 0:
        eps_local = epsilon0 * epsilon_Ez[1:nx, 1:ny, 0:nz] + 1e-30
        sigma_local = sigma_Ez[1:nx, 1:ny, 0:nz]
        alpha = (sigma_local * dt) / (2.0 * eps_local)
        denom = 1.0 + alpha
        numer_factor = 1.0 - alpha

        # include H-derivative psi terms for Ez update
        dHy_dx = (Hy[1:nx, 1:ny, 0:nz] - Hy[0:nx-1, 1:ny, 0:nz]) / dx
        dHx_dy = (Hx[1:nx, 1:ny, 0:nz] - Hx[1:nx, 0:ny-1, 0:nz]) / dx

        if psi_hy_dx is not None:
            psi_hy_dx = b_hy_dx * psi_hy_dx + c_hy_dx * dHy_dx
            dHy_eff3 = dHy_dx / kappa_hy_dx + psi_hy_dx
        else:
            dHy_eff3 = dHy_dx
        if psi_hx_dy is not None:
            psi_hx_dy = b_hx_dy * psi_hx_dy + c_hx_dy * dHx_dy
            dHx_eff3 = dHx_dy / kappa_hx_dy + psi_hx_dy
        else:
            dHx_eff3 = dHx_dy

        curlHz_eff = dHy_eff3 - dHx_eff3
        rhs = (curlHz_eff - Jz[1:nx, 1:ny, 0:nz]) * (dt / eps_local)
        Ez[1:nx, 1:ny, 0:nz] = (numer_factor * Ez[1:nx, 1:ny, 0:nz] + rhs) / denom
    #---- END UPDATE_E ----


    # apply simple E damping in PML
    if dampE is not None:
        # Ex/Ey/Ez have different shapes; apply component masks if available
        if dampE_Ex is not None:
            Ex *= dampE_Ex
        else:
            Ex *= dampE

        if dampE_Ey is not None:
            Ey *= dampE_Ey
        else:
            Ey *= dampE

        if dampE_Ez is not None:
            Ez *= dampE_Ez
        else:
            Ez *= dampE

    return (
        Hx, Hy, Hz,
        Ex, Ey, Ez,

        psi_ex_dy, psi_ex_dz,
        psi_ey_dx, psi_ey_dz,
        psi_ez_dx, psi_ez_dy,

        psi_hx_dy, psi_hx_dz,
        psi_hy_dx, psi_hy_dz,
        psi_hz_dx, psi_hz_dy
    )
