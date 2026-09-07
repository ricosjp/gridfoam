"""Unit tests for linear solver statistics."""

from __future__ import annotations

import torch

from gridfoam.core.equation import Equation, equation
from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.meta.config import SolverConfig
from gridfoam.meta.enums import (
    FieldRole,
    NormType,
    PreconditionerType,
    SolverType,
)
from gridfoam.solvers.cg import CGSolver


def _identity_equation(grid: AxisProjectedGrid, *, name: str = "p") -> Equation:
    p = CellField(grid, name, role=FieldRole.LOCAL, component_shape=())
    fv_matrix = FvMatrix(p)
    fv_matrix.diag.fill_(1.0)
    fv_matrix.upper.zero_()
    fv_matrix.lower.zero_()
    torch.manual_seed(0)
    fv_matrix.source = torch.randn(
        grid.num_cells,
        dtype=grid.dtype,
        device=grid.device,
    )
    p.data.zero_()
    return equation(p, fv_matrix)


def test_cg_returns_solver_stats(
    small_axis_projected_grid: AxisProjectedGrid,
) -> None:
    eq = _identity_equation(small_axis_projected_grid)
    solver = CGSolver(
        SolverConfig(
            method=SolverType.CG,
            preconditioner=PreconditionerType.NONE,
            tolerance=1e-10,
            rel_tolerance=0.0,
            max_iter=50,
            norm_type=NormType.L_2,
        )
    )
    result = solver.solve(eq)

    assert result.solution.shape == eq.target.data.shape
    assert len(result.stats) == 1
    assert result.stats[0].solver == SolverType.CG.value
    assert result.stats[0].iterations >= 0
    assert result.stats[0].initial_residual >= 0.0
    assert result.stats[0].final_residual >= 0.0
    assert result.stats[0].converged is True


def test_cg_returns_per_component_stats(
    small_axis_projected_grid: AxisProjectedGrid,
) -> None:
    grid = small_axis_projected_grid
    u = CellField(grid, "U", role=FieldRole.LOCAL, component_shape=(3,))
    fv_matrix = FvMatrix(u)
    fv_matrix.diag.fill_(1.0)
    fv_matrix.upper.zero_()
    fv_matrix.lower.zero_()
    torch.manual_seed(0)
    fv_matrix.source = torch.randn(
        grid.num_cells,
        3,
        dtype=grid.dtype,
        device=grid.device,
    )
    u.data.zero_()
    eq = equation(u, fv_matrix)

    solver = CGSolver(
        SolverConfig(
            method=SolverType.CG,
            preconditioner=PreconditionerType.NONE,
            tolerance=1e-10,
            rel_tolerance=0.0,
            max_iter=50,
            norm_type=NormType.L_2,
        )
    )
    result = solver.solve(eq)

    assert result.solution.shape == eq.target.data.shape
    assert len(result.stats) == 3
    assert all(stat.solver == SolverType.CG.value for stat in result.stats)
    assert all(stat.converged for stat in result.stats)
