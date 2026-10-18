#!/usr/bin/env python3
"""Run general_tests as a category.

Installs the optimization-off pgdl.so first. Scheduling tests keep their own
install path and are not started from here.
"""
from tools.general_extension import ensure_opt_off
from general_tests.verify_supported_tests import main as smoke_main

if __name__ == "__main__":
    ensure_opt_off()
    raise SystemExit(smoke_main())
