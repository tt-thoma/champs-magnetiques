import pickle
import traceback
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import TextTestResult

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
