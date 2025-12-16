import unittest

import numpy as np

from champs_v4.fdtd_yee_3d import Yee3D


class TestYee3D(unittest.TestCase):
    """Unit tests for Yee3D FDTD solver."""

    def test_initialization(self):
        """Test basic initialization."""
        nx, ny, nz = 10, 10, 10
        dx = 1e-3
        dt = dx / (3e8 * np.sqrt(3)) * 0.9  # Safe CFL
        sim = Yee3D(nx, ny, nz, dx, dt)

        self.assertEqual(sim.nx, nx)
        self.assertEqual(sim.ny, ny)
        self.assertEqual(sim.nz, nz)
        self.assertEqual(sim.dx, dx)
        self.assertEqual(sim.dt, dt)

        # Check field shapes
        self.assertEqual(sim.Ex.shape, (nx, ny + 1, nz + 1))
        self.assertEqual(sim.Ey.shape, (nx + 1, ny, nz + 1))
        self.assertEqual(sim.Ez.shape, (nx + 1, ny + 1, nz))
        self.assertEqual(sim.Hx.shape, (nx + 1, ny, nz))
        self.assertEqual(sim.Hy.shape, (nx, ny + 1, nz))
        self.assertEqual(sim.Hz.shape, (nx, ny, nz + 1))

        # Check materials
        self.assertEqual(sim.epsilon_r.shape, (nx, ny, nz))
        self.assertTrue(np.allclose(sim.epsilon_r, 1.0))
        self.assertTrue(np.allclose(sim.sigma, 0.0))

    def test_set_materials(self):
        """Test material setting."""
        nx, ny, nz = 5, 5, 5
        dx = 1e-3
        dt = dx / (3e8 * np.sqrt(3)) * 0.9
        sim = Yee3D(nx, ny, nz, dx, dt)

        epsilon_r = np.full((nx, ny, nz), 4.0)
        sigma = np.full((nx, ny, nz), 1e-3)
        sim.set_materials(epsilon_r, sigma)

        self.assertTrue(np.allclose(sim.epsilon_r, 4.0))
        self.assertTrue(np.allclose(sim.sigma, 1e-3))

        # Test invalid shapes
        with self.assertRaises(AssertionError):
            sim.set_materials(np.ones((nx + 1, ny, nz)))

    def test_step_stability(self):
        """Test that step() runs without crashing and fields change."""
        nx, ny, nz = 20, 20, 1  # 2D for speed
        dx = 1e-3
        dt = dx / (3e8 * np.sqrt(2)) * 0.9
        sim = Yee3D(nx, ny, nz, dx, dt)

        # Store initial fields
        Ex0 = sim.Ex.copy()
        Ey0 = sim.Ey.copy()
        Ez0 = sim.Ez.copy()

        # Run a few steps
        for _ in range(10):
            sim.step()

        # Fields should have changed (unless perfectly symmetric)
        self.assertTrue(not np.allclose(sim.Ex, Ex0) or np.allclose(sim.Ex, 0))
        self.assertTrue(not np.allclose(sim.Ey, Ey0) or np.allclose(sim.Ey, 0))
        self.assertTrue(not np.allclose(sim.Ez, Ez0) or np.allclose(sim.Ez, 0))

        # Check finiteness
        self.assertTrue(np.isfinite(sim.Ex).all())
        self.assertTrue(np.isfinite(sim.Ey).all())
        self.assertTrue(np.isfinite(sim.Ez).all())
        self.assertTrue(np.isfinite(sim.Hx).all())
        self.assertTrue(np.isfinite(sim.Hy).all())
        self.assertTrue(np.isfinite(sim.Hz).all())

    def test_plane_wave_propagation(self):
        """Test basic field updates."""
        nx, ny, nz = 10, 1, 1
        dx = 1e-3
        c = 3e8
        dt = dx / c * 0.9
        sim = Yee3D(nx, ny, nz, dx, dt)

        # Inject some energy
        sim.Ez[5, 0, 0] = 1.0

        # Run a few steps
        for _ in range(5):
            sim.step()

        # Fields should have changed
        self.assertGreater(np.sum(np.abs(sim.Ez)), 0)
        self.assertGreater(np.sum(np.abs(sim.Hx)), 0)

    def test_coil_addition(self):
        """Test adding a coil source."""
        nx, ny, nz = 20, 20, 20
        dx = 1e-3
        dt = dx / (3e8 * np.sqrt(3)) * 0.9
        sim = Yee3D(nx, ny, nz, dx, dt)

        center = (nx // 2, ny // 2, nz // 2)
        radius = 3
        sim.add_coil(center, radius, axis="z", turns=1, current=1.0)

        # Check that Jz has non-zero values in the coil region
        self.assertGreater(np.sum(np.abs(sim.Jz)), 0)

    def test_pml_initialization(self):
        """Test PML setup."""
        nx, ny, nz = 10, 10, 10
        dx = 1e-3
        dt = dx / (3e8 * np.sqrt(3)) * 0.9
        pml_width = 3
        sim = Yee3D(nx, ny, nz, dx, dt, pml_width=pml_width)

        # Check that damping arrays exist and are shaped correctly
        self.assertTrue(hasattr(sim, "dampE_Ex"))
        self.assertEqual(sim.dampE_Ex.shape, sim.Ex.shape)
        self.assertTrue(np.all(sim.dampE_Ex <= 1.0))  # Damping factors <= 1


if __name__ == "__main__":
    unittest.main()
