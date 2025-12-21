import datetime
import os
import pickle
import subprocess
import sys
import warnings
from optparse import Values
from pathlib import Path
from unittest import (
    TestSuite,
    TextTestResult,
    TextTestRunner,
    defaultTestLoader,
)

from ._options import opts
from ._result import TIMINGS, GitHubTestResult
from .test_cache import TestCache
from .test_CFL_check import TestCFLCheck
from .test_dispersion import TestDispersion
from .test_examples import TestExamples
from .test_physical_consistency import TestPhysicalConsistency
from .test_world_basic import TestWorldBasic
from .test_yee3d import TestYee3D
from .test_yee_plane_wave_3d import TestYeePlaneWave3D
from .test_yee_skin_depth import TestYeeSkinDepth

RESULTS: Path = Path("./tests/results")
RESULTS.mkdir(parents=True, exist_ok=True)


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


def custom(message, category, filename, lineno, _file=None, _line=None) -> None:
    """Function to format a warning the standard way."""
    sys.stdout.write(
        f"\n::warning file={filename},line={lineno},title={category.__name__}::{message}\n"
    )


if __name__ == "__main__":
    warnings.simplefilter("always")
    warnings.showwarning = custom  # ty: ignore

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
        if not opts.local:
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
        if not opts.local:
            subprocess.run(["git", "push", "-u", "origin", "results"], check=True)
        next_commit: str = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, check=True
            )
            .stdout.strip()
            .decode()
        )

        print(f"{prev_commit=} {next_commit=}")
        URL = (
            "https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/{0}/examples/results/"
            "{1}/{2}"
        )
        summary += "\n# Results\n\n"
        for subdir in Path("./examples/results/").iterdir():
            folder: str = subdir.name
            summary += f"## {folder}\n\n"
            for subfile in subdir.iterdir():
                if subfile.is_file():
                    image: str = subfile.name
                    summary += f"### {image}\n\n"
                    summary += "| Before | After |\n| --- | --- |\n"
                    summary += (
                        f"| ![Before]({URL.format(prev_commit, folder, image)})"
                        f"| ![After]({URL.format(next_commit, folder, image)})"
                        "|\n\n"
                    )

        subprocess.run(["git", "checkout", "master"], check=True)

        with open(
            os.environ.get("GITHUB_STEP_SUMMARY", RESULTS / "summary.md"), "w"
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
