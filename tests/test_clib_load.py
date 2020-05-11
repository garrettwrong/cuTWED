#!/usr/bin/env python
"""
Confirm the C library is found and can load.

Copyright 2020 Garrett Wright, Gestalt Group LLC
"""

import ctypes


def test_library_load():
    _ = ctypes.CDLL('libcuTWED.so')
