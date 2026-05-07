"""Unit tests for ``gridfoam.solvers.factory``."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gridfoam.meta.config import SolverConfig
from gridfoam.meta.enums import SolverType
from gridfoam.solvers.bicgstab import BiCGSTABSolver
from gridfoam.solvers.cg import CGSolver
from gridfoam.solvers.factory import create_solver
from gridfoam.solvers.pyamg_bridge import PyamgBridgeSolver


def test_create_solver_returns_cg():
    cfg = SolverConfig(method=SolverType.CG)
    assert isinstance(create_solver(cfg), CGSolver)


def test_create_solver_returns_bicgstab():
    cfg = SolverConfig(method=SolverType.BiCGSTAB)
    assert isinstance(create_solver(cfg), BiCGSTABSolver)


def test_create_solver_returns_pyamg():
    cfg = SolverConfig(method=SolverType.PyAMG)
    assert isinstance(create_solver(cfg), PyamgBridgeSolver)


def test_create_solver_unknown_method_raises():
    cfg = MagicMock()
    cfg.method = "not_a_solver"
    with pytest.raises(ValueError, match="Unknown solver method"):
        create_solver(cfg)
