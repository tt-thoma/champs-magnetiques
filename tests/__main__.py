import sys
import traceback
import unittest
import warnings
from optparse import OptionParser, Values
from typing import TYPE_CHECKING
from unittest import (
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

if TYPE_CHECKING:
    from _typeshed import OptExcInfo


class GitHubTestResult(TextTestResult):
    def startTest(self, test: unittest.case.TestCase) -> None:
        self.stream.write(f"::group::{self.getDescription(test)}\n")
        self.stream.flush()
        super().startTest(test)
        self.stream.flush()

    def addError(self, test: unittest.case.TestCase, err: "OptExcInfo") -> None:
        if err[1] is not None and err[1].__traceback__ is not None:
            pretty_err: str = traceback.format_exception(err[1])[-1].strip("\n")
            frame: traceback.FrameSummary = traceback.extract_tb(err[1].__traceback__)[
                -1
            ]
            self.stream.write(
                f"\n::error file={frame.filename},line={frame.lineno},endLine={frame.end_lineno},col={frame.colno},"
                f"endCol={frame.end_colno},title={str(test)}::{pretty_err}\n"
            )
        else:
            self.stream.write(f"\n::error title={str(test)}::ERROR\n")
        super().addError(test, err)
        self.stream.write("::endgroup::\n")
        self.stream.flush()

    def addSuccess(self, test: unittest.case.TestCase) -> None:
        super().addSuccess(test)
        self.stream.write("::endgroup::\n")
        self.stream.flush()

    def addSkip(self, test: unittest.case.TestCase, reason: str) -> None:
        self.stream.write(f"\n::notice title={str(test)}::{reason}\n")
        super().addSkip(test, reason)
        self.stream.write("::endgroup::\n")
        self.stream.flush()


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


def custom(message, category, filename, lineno, line=None) -> str:
    """Function to format a warning the standard way."""
    return f"\n::warning file={filename},line={lineno},title={category.__name__}::{message}\n"


if __name__ == "__main__":
    warnings.simplefilter("always")
    warnings.formatwarning = custom  # ty: ignore

    parser: OptionParser = OptionParser()
    parser.add_option("-f", "--full", action="store_true", dest="full", default=False)
    parser.add_option(
        "-g",
        "--github",
        action="store_const",
        dest="resultclass",
        default=None,
        const=GitHubTestResult,
    )
    opts: Values
    args: list[str]
    opts, args = parser.parse_args()

    runner: TextTestRunner = TextTestRunner(
        verbosity=2,
        durations=0,
        resultclass=opts.resultclass,
        warnings="always",
        stream=sys.stdout,
    )
    results: TextTestResult = runner.run(test_suite(opts))

    if not results.wasSuccessful():
        sys.exit(-1)

    # case_name: str
    # duration: int | float
    # for case_name, duration in results.collectedDurations:
    #     print(f"{case_name}: {duration}")
