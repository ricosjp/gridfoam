"""Unit tests for ``AlgorithmBase.solvers``."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.helpers import simple_convergence_config

from gridfoam.algorithms.base import AlgorithmBase
from gridfoam.algorithms.simple import SIMPLE
from gridfoam.core.grid.factory import create_grid
from gridfoam.solvers.base import LinearSolver


def test_algorithm_base_cannot_be_instantiated() -> None:
    # AlgorithmBase remains abstract.
    with pytest.raises(TypeError):
        AlgorithmBase()  # pyright: ignore[reportAbstractUsage]


def test_simple_solvers_match_fvsolution(tmp_path: Path) -> None:
    # solvers is a dict of LinearSolver keyed by fvSolution field names.
    grid = create_grid(simple_convergence_config(tmp_path))
    algo = SIMPLE(grid)
    assert isinstance(algo.solvers, dict)
    assert set(algo.solvers) == set(grid.sim_config.fvSolution.solvers)
    assert all(
        isinstance(solver, LinearSolver) for solver in algo.solvers.values()
    )


def test_simple_solver_grad_mode_is_mutable(tmp_path: Path) -> None:
    # Mutating solver.grad_mode updates the same solver object.
    grid = create_grid(simple_convergence_config(tmp_path))
    algo = SIMPLE(grid)
    name = next(iter(algo.solvers))
    solver = algo.solvers[name]
    solver.grad_mode = "unrolled"
    assert algo.solvers[name] is solver
    assert algo.solvers[name].grad_mode == "unrolled"


def test_set_grad_mode_updates_all_solvers(tmp_path: Path) -> None:
    # set_grad_mode writes through to every registered solver.
    grid = create_grid(simple_convergence_config(tmp_path))
    algo = SIMPLE(grid)
    algo.set_grad_mode("unrolled")
    assert all(
        solver.grad_mode == "unrolled" for solver in algo.solvers.values()
    )
