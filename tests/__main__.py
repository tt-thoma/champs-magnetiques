import inspect
import sys
from optparse import OptionParser, Values
from unittest import (
    TestCase,
    TestSuite,
    TextTestResult,
    TextTestRunner,
    defaultTestLoader,
)

from .test_CFL_check import TestCFLCheck
from .test_dispersion import TestDispersion
from .test_examples import TestExamples
from .test_physical_consistency import TestPhysicalConsistency
from .test_world_basic import TestWorldBasic
from .test_yee3d import TestYee3D
from .test_yee_plane_wave_3d import TestYeePlaneWave3D
from .test_yee_skin_depth import TestYeeSkinDepth


def test_suite(options: Values) -> TestSuite:
    suite: TestSuite = TestSuite()

    suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestDispersion))
    suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestYee3D))
    suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestYeePlaneWave3D))
    suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestYeeSkinDepth))

    if options.full:
        suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestCFLCheck))
        suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestExamples))
        suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestPhysicalConsistency))
        suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestWorldBasic))

    return suite


if __name__ == "__main__":
    parser: OptionParser = OptionParser()
    parser.add_option("-f", "--full", action="store_true", dest="full", default=False)
    opts: Values
    args: list[str]
    opts, args = parser.parse_args()

    runner: TextTestRunner = TextTestRunner(verbosity=2, durations=0)
    results: TextTestResult = runner.run(test_suite(opts))

    case: TestCase
    reason: str
    for case, reason in results.skipped:
        print(
            f"::notice file={inspect.getfile(case.__class__)},col={inspect.getsourcelines(case)[1]},"
            f"title={str(case)} ... skipped::{reason}"
        )

    if not results.wasSuccessful():
        sys.exit(-1)

    # case_name: str
    # duration: int | float
    # for case_name, duration in results.collectedDurations:
    #     print(f"{case_name}: {duration}")
