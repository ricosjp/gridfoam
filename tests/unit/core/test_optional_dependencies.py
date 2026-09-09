"""Optional dependencies do not prevent importing gridfoam's core APIs."""

import subprocess
import sys


def test_core_import_does_not_require_graphlow() -> None:
    """Core APIs remain importable when graphlow cannot be resolved."""
    code = """
import importlib.abc
import sys


class BlockGraphlow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "graphlow" or fullname.startswith("graphlow."):
            raise ModuleNotFoundError("blocked graphlow")
        return None


sys.meta_path.insert(0, BlockGraphlow())
from gridfoam.core import GridBase, create_grid
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
