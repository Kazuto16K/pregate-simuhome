"""Import helper for tests that need SimuHome AND this project's code.

Both use a top-level package called `src`. As on Kaggle, this project's package is
loaded under the name `irlib` (its modules use relative imports, so the name does
not matter), and SimuHome's checkout goes on sys.path so `src.*` is SimuHome's.
"""

from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)


def setup(simuhome_dir: str):
    simuhome_dir = os.path.abspath(simuhome_dir)
    if simuhome_dir not in sys.path:
        sys.path.insert(0, simuhome_dir)
    if "irlib" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "irlib", os.path.join(PROJECT, "src", "__init__.py"),
            submodule_search_locations=[os.path.join(PROJECT, "src")],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["irlib"] = mod
        spec.loader.exec_module(mod)
    return simuhome_dir
