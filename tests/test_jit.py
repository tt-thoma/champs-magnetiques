import time
from unittest import TestCase, main

from champs_v4.fdtd_yee_3d import NumbaYee3D, Yee3D
from examples.run_simulation import main as run_simulation


class TestJIT(TestCase):
    def test_jit(self) -> None:
        start: int = time.perf_counter_ns()
        run_simulation(Yee3D)
        default_time: int = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        run_simulation(NumbaYee3D)
        numba_time: int = time.perf_counter_ns() - start
        self.assertLess(numba_time, default_time, "Numba optimized function is slower than standard "
            f"({default_time} ns (default) < {numba_time} ns (numba))")


if __name__ == "__main__":
    main()
