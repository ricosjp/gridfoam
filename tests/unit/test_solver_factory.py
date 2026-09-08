"""
Solver methods select their implementation; invalid configs fail runtime
checking.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from gridfoam.meta.config import SolverConfig
from gridfoam.meta.enums import SolverType
from gridfoam.solvers.bicgstab import BiCGSTABSolver
from gridfoam.solvers.cg import CGSolver
from gridfoam.solvers.factory import create_solver
from gridfoam.solvers.pyamg_bridge import PyamgBridgeSolver


def test_create_solver_returns_cg() -> None:
    """CG method maps to ``CGSolver``."""
    cfg = SolverConfig(method=SolverType.CG)
    assert isinstance(create_solver(cfg), CGSolver)


def test_create_solver_returns_bicgstab() -> None:
    """BiCGSTAB method maps to ``BiCGSTABSolver``."""
    cfg = SolverConfig(method=SolverType.BiCGSTAB)
    assert isinstance(create_solver(cfg), BiCGSTABSolver)


def test_create_solver_returns_pyamg() -> None:
    """PyAMG method maps to ``PyamgBridgeSolver``."""
    cfg = SolverConfig(method=SolverType.PyAMG)
    assert isinstance(create_solver(cfg), PyamgBridgeSolver)


def test_create_solver_unknown_method_raises() -> None:
    """Invalid method values must be rejected by runtime type checking."""
    cfg = MagicMock()
    cfg.method = "not_a_solver"
    with pytest.raises(BeartypeCallHintParamViolation):
        create_solver(cfg)
