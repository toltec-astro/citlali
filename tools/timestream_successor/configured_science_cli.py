#!/usr/bin/env python3
"""Explicit single-observation compact-source SCIENCE entry.

The shared coordinator retains the accepted native observation lifetime and
resource ownership. Its mode admission rejects POINT requests at this entry.
"""

from configured_point_cli import execute as _execute, main as _main


def execute(config_path, native, *, launch_dir=None):
    return _execute(config_path, native, launch_dir=launch_dir, mode="science")


def main(argv=None):
    return _main(argv, mode="science")


if __name__ == "__main__":
    raise SystemExit(main())
