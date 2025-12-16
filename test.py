#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import sys
import logging

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO
)

import examples.run_coil as t

t.main()
