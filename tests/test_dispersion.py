import unittest

import numpy as np
import matplotlib.pyplot as plt


class TestDispersion(unittest.TestCase):
    def test_plane_wave_dispersion(self):
        """
        Test plane wave propagation dispersion analytically.
        """
        # Parameters
        dx = 1e-3  # 1 mm
        c = 3e8
        dt = dx / c * 0.9  # CFL safe

        # Frequencies to test
        frequencies = [1e5, 5e5, 1e6, 5e6, 10e6]  # 0.1 to 10 MHz
        k_numerical = []

        for f in frequencies:
            omega = 2 * np.pi * f
            k_exact = omega / c

            # Numerical dispersion relation for 1D FDTD:
            # cos(omega * dt) = 1 - 2 * (c * dt / dx)^2 * sin^2(k * dx / 2)
            # Solve for k

            # Define the function to solve: cos(omega dt) - 1 + 2 (c dt / dx)^2 sin^2(k dx / 2) = 0
            from scipy.optimize import fsolve

            def dispersion_eq(k):
                return np.cos(omega * dt) - 1 + 2 * (c * dt / dx)**2 * np.sin(k * dx / 2)**2

            # Initial guess k = omega / c
            k_guess = k_exact
            k_num = fsolve(dispersion_eq, k_guess)[0]

            print(f"Frequency {f/1e6} MHz: k_exact = {k_exact:.4f}, k_num = {k_num:.4f}, error = {abs(k_num - k_exact)/k_exact * 100:.2f}%")

            k_numerical.append(k_num)

        # Plot
        k_exact_list = [2*np.pi*f / c for f in frequencies]
        k_num_list = k_numerical

        plt.figure()
        plt.plot(frequencies, k_exact_list, 'o-', label='Exact')
        plt.plot(frequencies, k_num_list, 's-', label='Numerical')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Wavenumber k (rad/m)')
        plt.title('Dispersion Relation: Exact vs Numerical')
        plt.legend()
        plt.grid(True)
        plt.savefig('tests/results/dispersion_test.png', dpi=150)
        #plt.show()  # Don't show plot since it slows down tests

        print("Dispersion test completed. Plot saved as dispersion_test.png")


if __name__ == '__main__':
    unittest.main()
