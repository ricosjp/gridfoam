"""
Final solver keys take priority, missing final keys fall back, and missing
fields fail.
"""

from __future__ import annotations

import pytest

from gridfoam.meta.config import SolverConfig
from gridfoam.meta.enums import SolverType
from gridfoam.solvers.factory import create_solver
from gridfoam.solvers.resolver import resolve_solver, resolve_solver_config


def test_resolve_solver_config_prefers_final_key() -> None:
    """Final-step solve must pick ``pFinal`` over ``p`` when both exist."""
    solvers = {
        "p": SolverConfig(method=SolverType.CG, tolerance=1e-6),
        "pFinal": SolverConfig(method=SolverType.CG, tolerance=1e-8),
    }
    cfg = resolve_solver_config(solvers, "p", is_final=True)
    assert cfg.tolerance == 1e-8


def test_resolve_solver_config_falls_back_to_primary() -> None:
    """Without a Final key, the primary solver config is used."""
    solvers = {"p": SolverConfig(method=SolverType.BiCGSTAB)}
    cfg = resolve_solver_config(solvers, "p", is_final=True)
    assert cfg.method is SolverType.BiCGSTAB


def test_resolve_solver_runtime_instance() -> None:
    """Runtime resolver returns the instantiated solver object, not config."""
    solvers = {
        "p": create_solver(SolverConfig(method=SolverType.CG)),
        "p_final": create_solver(
            SolverConfig(method=SolverType.BiCGSTAB, tolerance=1e-9)
        ),
    }
    solver = resolve_solver(solvers, "p", is_final=True)
    assert solver is solvers["p_final"]


def test_resolve_solver_config_missing_key_raises() -> None:
    """Missing solver keys must raise ``KeyError``."""
    with pytest.raises(KeyError):
        resolve_solver_config({}, "p")
