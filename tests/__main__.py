from unittest import TestSuite, TextTestRunner
from unittest import defaultTestLoader

from .test_yee_skin_depth import TestYeeSkinDepth
from .test_yee_plane_wave_3d import TestYeePlaneWave3D
from .test_yee3d import TestYee3D
from .test_dispersion import TestDispersion


def test_suite() -> TestSuite:
    suite: TestSuite = TestSuite()
    suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestDispersion))
    suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestYee3D))
    suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestYeePlaneWave3D))
    suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestYeeSkinDepth))
    return suite


if __name__ == '__main__':
    runner = TextTestRunner(verbosity=1)
    runner.run(test_suite())
