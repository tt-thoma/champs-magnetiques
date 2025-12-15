from unittest import TestCase, skip

from examples.run_antenna_precise_long import main as run_antenna_precise_long
from examples.run_antenna_test import main as run_antenna_test
from examples.run_coil import main as run_coil
from examples.run_coil_anim import main as run_coil_anim
from examples.run_coil_slice_anim import main as run_coil_slice_anim
from examples.run_simulation import main as run_simulation


class TestExamples(TestCase):
    @skip("Takes too long")
    def test_run_antenna_precise_long(self) -> None:
        run_antenna_precise_long()

    def test_run_antenna_test(self) -> None:
        run_antenna_test()

    def test_run_coil(self) -> None:
        run_coil()

    def test_run_coil_anim(self) -> None:
        run_coil_anim()

    def test_run_coil_slice_anim(self) -> None:
        run_coil_slice_anim()

    def test_run_simulation(self) -> None:
        run_simulation()
