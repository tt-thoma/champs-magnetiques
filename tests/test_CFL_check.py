from unittest import main, skip, TestCase


@skip("Ancienne simulation")
class TestCFLCheck(TestCase):
    def test_cfl(self) -> None:
        import sys
        sys.path.insert(0, r'c:\Espace de programation\champs-magnetiques\champs_v4')
        from world import World
        import constants as const
        import math
        size = 0.2
        cell_size = 0.05
        # choose a dt deliberately too large
        dt = 1.0
        w = World(size, cell_size, dt)
        # compute reference CFL
        c = const.c
        cfl = cell_size / (c * math.sqrt(3.0))
        self.assertLessEqual(w.dt, 0.9 * cfl + 1e-20, f"dt was not reduced correctly: {w.dt} > 0.9*CFL")
        print(f"CFL test passed: dt adjusted to {w.dt:.3e} (0.9*CFL = {0.9*cfl:.3e})")


if __name__ == '__main__':
    main()
