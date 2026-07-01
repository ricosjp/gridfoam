"""Resolve OpenFOAM-style solver keys such as ``pFinal``."""

from __future__ import annotations

from gridfoam.meta.config import SolverConfig
from gridfoam.solvers.base import LinearSolver


def resolve_solver_config(
    solvers: dict[str, SolverConfig],
    field_name: str,
    *,
    is_final: bool = False,
) -> SolverConfig:
    """
    Resolve a solver configuration with OpenFOAM ``*Final`` key fallback.

    Parameters
    ----------
    solvers : dict[str, SolverConfig]
        Solver configuration table from ``fvSolution``.
    field_name : str
        Primary field name such as ``p`` or ``Phi``.
    is_final : bool, optional
        When ``True``, prefer ``{field_name}Final`` or ``{field_name}_final``.

    Returns
    -------
    SolverConfig
        Matching solver configuration.

    Raises
    ------
    KeyError
        If no matching solver configuration exists.
    """
    if is_final:
        for key in (f"{field_name}Final", f"{field_name}_final"):
            if key in solvers:
                return solvers[key]
    if field_name not in solvers:
        raise KeyError(f"Solver configuration for {field_name!r} not found")
    return solvers[field_name]


def resolve_solver(
    solvers: dict[str, LinearSolver],
    field_name: str,
    *,
    is_final: bool = False,
) -> LinearSolver:
    """
    Resolve a runtime linear solver with OpenFOAM ``*Final`` key fallback.

    Parameters
    ----------
    solvers : dict[str, LinearSolver]
        Runtime solver instances keyed by field name.
    field_name : str
        Primary field name such as ``p`` or ``Phi``.
    is_final : bool, optional
        When ``True``, prefer ``{field_name}Final`` or ``{field_name}_final``.

    Returns
    -------
    LinearSolver
        Matching linear solver instance.

    Raises
    ------
    KeyError
        If no matching solver instance exists.
    """
    if is_final:
        for key in (f"{field_name}Final", f"{field_name}_final"):
            if key in solvers:
                return solvers[key]
    if field_name not in solvers:
        raise KeyError(f"Solver for {field_name!r} not found")
    return solvers[field_name]
