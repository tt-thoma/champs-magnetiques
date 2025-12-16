from unittest import TestCase, main, skip


@skip("Ancienne simulation")
class TestPhysicalConsistency(TestCase):
    def test_physical(self) -> None:
        import sys

        import numpy as np

        sys.path.insert(0, r"c:\Espace de programation\champs-magnetiques\champs_v4")
        import constants as const
        from corps import Particle
        from world import World

        def curl_field(field, cell_size):
            # compute curl for vector field with centered differences and periodicity
            Fx = field[..., 0]
            Fy = field[..., 1]
            Fz = field[..., 2]
            dx = cell_size
            curl_x = (np.roll(Fz, -1, axis=1) - np.roll(Fz, 1, axis=1)) / (2 * dx) - (
                np.roll(Fy, -1, axis=2) - np.roll(Fy, 1, axis=2)
            ) / (2 * dx)
            curl_y = (np.roll(Fx, -1, axis=2) - np.roll(Fx, 1, axis=2)) / (2 * dx) - (
                np.roll(Fz, -1, axis=0) - np.roll(Fz, 1, axis=0)
            ) / (2 * dx)
            curl_z = (np.roll(Fy, -1, axis=0) - np.roll(Fy, 1, axis=0)) / (2 * dx) - (
                np.roll(Fx, -1, axis=1) - np.roll(Fx, 1, axis=1)
            ) / (2 * dx)
            return np.stack((curl_x, curl_y, curl_z), axis=-1)

        # small domain
        size = 0.2
        cell_size = 0.05
        dt = 0.001

        w = World(size, cell_size, dt, I=0.0, U=0.0, duree_simulation=0.05)

        # add a particle off-center so J may be nonzero when moving
        center = size / 2
        p = Particle(
            center + 0.01, center, center, const.charge_electron, const.masse_electron
        )
        p.vx = 0.0
        p.vy = 0.0
        p.vz = 0.0
        w.add_part(p)

        # Save fields
        B_before = w.field_B.copy()
        E_before = w.field_E.copy()

        # Compute J used by FDTD
        J = w.compute_current_density_J()

        # Update B using implemented method (this mutates field_B)
        w.update_B_FDTD()
        B_after = w.field_B.copy()

        # Check Faraday: (B_after - B_before)/dt + curl(E_before) == 0
        dBdt = (B_after - B_before) / w.dt
        curlE = curl_field(E_before, w.cell_size)
        res_faraday = dBdt + curlE
        max_faraday = np.max(np.abs(res_faraday))

        # Update E using Ampere-Maxwell
        w.update_E_FDTD(J)
        E_after = w.field_E.copy()

        # Check Ampere-Maxwell: (E_after - E_before)/dt - (1/epsilon0)*(curl(B_after) - J) == 0
        dEdt = (E_after - E_before) / w.dt
        curlB = curl_field(B_after, w.cell_size)
        rhs = (1.0 / const.epsilon_0) * (curlB - J)
        res_ampe = dEdt - rhs
        max_ampe = np.max(np.abs(res_ampe))

        print(f"Max Faraday residual: {max_faraday:.3e}")
        print(f"Max Ampere-Maxwell residual: {max_ampe:.3e}")

        # Tolerances are loose because of discrete differencing and eps handling
        self.assertLess(max_faraday, 1e-6, f"Faraday residual too large: {max_faraday}")
        self.assertLess(
            max_ampe, 1e-6, f"Ampere-Maxwell residual too large: {max_ampe}"
        )
        print("Physical consistency test passed")


if __name__ == "__main__":
    main()
