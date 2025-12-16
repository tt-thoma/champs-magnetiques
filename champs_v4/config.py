#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# SenfinaLight, VirinasCode, tt_thoma

import functools
import importlib.util
import logging
from curses import wrapper

import numpy as np

logger = logging.getLogger(__name__)

NUMBA = importlib.util.find_spec("numba") is not None
# NUMBA = False
if NUMBA:
    logger.info("Numba found. Automatically applying it.")

    import numba

    njit = numba.njit
    range_f = numba.prange
else:
    logger.info("Numba not found. Applying default methods/skipping decorator.")

    def njit(*_, **__):
        return lambda func: lambda *args, **kwargs: func(*args, **kwargs)

    range_f = range


int_t = int
float_t = np.float64
ndarray_t = np.ndarray[tuple[int, int, int], np.dtype]
# ndarray_t = np.ndarray[tuple[int_t, int_t, int_t], float_t]
