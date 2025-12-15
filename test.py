#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import sys

import examples.run_coil as t
import logging

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG
)

t.main()
