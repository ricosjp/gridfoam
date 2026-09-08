"""Loose residual thresholds let SIMPLE report convergence after one step."""

from __future__ import annotations

import pathlib

from tests.helpers import simple_convergence_config

from gridfoam.algorithms.simple import SIMPLE
from gridfoam.core.grid.factory import create_grid


def test_simple_has_converged_with_loose_residual_control(
    tmp_path: pathlib.Path,
) -> None:
    """
    With very loose residualControl thresholds, one SIMPLE step must mark
    the algorithm as converged immediately.
    """
    grid = create_grid(simple_convergence_config(tmp_path))
    algo = SIMPLE(grid)
    algo.step()
    assert algo.has_converged()
