import datetime
import os
import pickle
import subprocess
import sys
import traceback
import unittest
import warnings
from optparse import OptionParser, Values
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import (
    TestSuite,
    TextTestResult,
    TextTestRunner,
    defaultTestLoader,
)

from .test_cache import TestCache
from .test_CFL_check import TestCFLCheck
from .test_dispersion import TestDispersion
from .test_examples import TestExamples
from .test_physical_consistency import TestPhysicalConsistency
from .test_world_basic import TestWorldBasic
from .test_yee3d import TestYee3D
from .test_yee_plane_wave_3d import TestYeePlaneWave3D
from .test_yee_skin_depth import TestYeeSkinDepth

if TYPE_CHECKING:
    from unittest.runner import _WritelnDecorator

    from _typeshed import OptExcInfo

TIMINGS: Path = Path("./tests/results/timings.dat")
timings: dict[str, list[float]]
if TIMINGS.exists():
    with open(TIMINGS, "rb") as timings_file:
        timings = pickle.load(timings_file)
else:
    timings = {}


class GitHubTestResult(TextTestResult):
    def __init__(
        self,
        stream: "_WritelnDecorator",
        descriptions: bool,
        verbosity: int,
        *,
        durations: int | None = None,
    ) -> None:
        super().__init__(stream, descriptions, verbosity, durations=durations)
        self.timings: dict[str, list[float]] = {}

    def startTest(self, test: unittest.case.TestCase) -> None:
        self.stream.write(f"::group::{self.getDescription(test)}\n")
        self.stream.flush()
        super().startTest(test)
        self.stream.flush()

    def __error(self, test: unittest.case.TestCase, err: "OptExcInfo") -> None:
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

    def addError(self, test: unittest.case.TestCase, err: "OptExcInfo") -> None:
        self.__error(test, err)
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

    def addFailure(self, test: unittest.case.TestCase, err: "OptExcInfo") -> None:
        self.__error(test, err)
        super().addFailure(test, err)
        self.stream.write("::endgroup::\n")
        self.stream.flush()

    def addDuration(self, test: unittest.case.TestCase, elapsed: float) -> None:
        self.timings[str(test)] = timings.get(str(test), []) + [elapsed]
        return super().addDuration(test, elapsed)


def test_suite(options: Values) -> TestSuite:
    suite: TestSuite = TestSuite()

    suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestDispersion))
    suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestYee3D))
    suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestYeePlaneWave3D))
    suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestYeeSkinDepth))
    suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestCache))

    if options.full:
        suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestCFLCheck))
        suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestExamples))
        suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestPhysicalConsistency))
        suite.addTests(defaultTestLoader.loadTestsFromTestCase(TestWorldBasic))

    return suite


def custom(message, category, filename, lineno, file=None, line=None) -> None:
    """Function to format a warning the standard way."""
    sys.stdout.write(
        f"\n::warning file={filename},line={lineno},title={category.__name__}::{message}\n"
    )


if __name__ == "__main__":
    warnings.simplefilter("always")
    warnings.showwarning = custom  # ty: ignore

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
        verbosity=4,
        durations=0,
        resultclass=opts.resultclass,
        warnings="always",
        stream=sys.stdout,
    )
    results: TextTestResult | GitHubTestResult = runner.run(test_suite(opts))

    # Summary file
    if isinstance(results, GitHubTestResult):
        summary: str = (
            "# Timings\n\n| Test | Previous | Latest | Improvement | Diff |\n"
            "| :--- | :---: | :---: | ---: | ---: |\n"
        )
        test_name: str
        test_time: list[float]
        for test_name, test_time in {
            k: v
            for k, v in sorted(
                results.timings.items(), key=lambda item: item[1], reverse=True
            )
        }.items():
            previous_time: float = test_time[-2] if len(test_time) > 1 else float("inf")
            improvement: float = (
                test_time[-1] / test_time[-2] if len(test_time) > 1 else 1
            )
            diff: float = (test_time[-1] - previous_time) / previous_time
            summary += (
                f"| {test_name} | {previous_time:.3f} s | {test_time[-1]:.3f} s | x{improvement:.3f} "
                f"| {diff:+.2%} |\n"
            )

        # Add images
        """
        git config user.name "github-actions[bot]"
        git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
        git checkout results
        git add examples/results/
        git commit --allow-empty -m "${{ github.run_number }}.${{ github.run_attempt }}"
        git push -u origin results
        git checkout master
        """
        subprocess.run(
            ["git", "config", "user.name", "github-actions[bot]"], check=True
        )
        subprocess.run(
            [
                "git",
                "config",
                "user.email",
                "41898282+github-actions[bot]@users.noreply.github.com",
            ],
            check=True,
        )
        subprocess.run(["git", "checkout", "results"], check=True)
        prev_commit: str = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, check=True
            )
            .stdout.strip()
            .decode()
        )
        subprocess.run(["git", "add", "examples/results/"], check=True)
        subprocess.run(
            [
                "git",
                "commit",
                "--allow-empty",
                "-m",
                datetime.datetime.now().isoformat(),
            ],
            check=True,
        )
        subprocess.run(["git", "push", "-u", "origin", "results"], check=True)
        next_commit: str = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, check=True
            )
            .stdout.strip()
            .decode()
        )
        subprocess.run(["git", "checkout", "master"], check=True)

        print(f"{prev_commit=} {next_commit=}")

        with open(
            os.environ.get("GITHUB_STEP_SUMMARY", "./tests/results/summary.md"), "w"
        ) as summary_file:
            summary_file.write(summary)
        with open(TIMINGS, "wb") as timings_file_w:
            pickle.dump(results.timings, timings_file_w)

    if not results.wasSuccessful():
        sys.exit(-1)

    # case_name: str
    # duration: int | float
    # for case_name, duration in results.collectedDurations:
    #     print(f"{case_name}: {duration}")
