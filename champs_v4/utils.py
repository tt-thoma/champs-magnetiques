"""
Helpers --- Shouldn't contain any math
"""

import cProfile
from typing import Callable
from functools import wraps


def profile[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    profiler: cProfile.Profile = cProfile.Profile()

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        returned: R = profiler.runcall(function, *args, **kwargs)
        profiler.dump_stats("restats")
        return returned

    return wrapped
