"""Strict step solves must propagate primal and delayed backward failures."""

import pytest
import torch

from gridfoam.core.equation import Equation, equation
from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.meta.config import SolverConfig
from gridfoam.meta.enums import FieldRole, SolverType
from gridfoam.solvers.base import (
    GradientMode,
    LinearSolveError,
    require_converged_solves,
)
from gridfoam.solvers.factory import create_solver


def make_equation(grid: AxisProjectedGrid, source: torch.Tensor) -> Equation:
    field = CellField(grid, "strict_solve", FieldRole.LOCAL, ())
    field.data = torch.zeros_like(source)
    matrix = FvMatrix(field)
    matrix.diag = torch.ones_like(source) * 2
    matrix.source = source
    return equation(field, matrix)


@pytest.mark.parametrize(
    "method", [SolverType.CG, SolverType.BiCGSTAB, SolverType.PyAMG]
)
@pytest.mark.parametrize("mode", ["adjoint", "unrolled"])
def test_primal_budget_exhaustion_raises_and_restores_policy(
    small_axis_projected_grid: AxisProjectedGrid,
    method: SolverType,
    mode: GradientMode,
) -> None:
    grid = small_axis_projected_grid
    solver = create_solver(SolverConfig(method=method, max_iter=0))
    solver.grad_mode = mode
    eq = make_equation(grid, torch.ones(grid.num_cells, dtype=grid.dtype))
    with pytest.raises(LinearSolveError, match="Primal"):
        with require_converged_solves([solver, solver]):
            solver.solve(eq)
    assert not solver.require_convergence
    # Ordinary standalone calls retain the established report-only behavior.
    assert not solver.solve(eq).stats[0].converged


@pytest.mark.parametrize(
    "method", [SolverType.CG, SolverType.BiCGSTAB, SolverType.PyAMG]
)
def test_backward_failure_survives_scope_exit_and_later_setting_changes(
    small_axis_projected_grid: AxisProjectedGrid, method: SolverType
) -> None:
    grid = small_axis_projected_grid
    solver = create_solver(SolverConfig(method=method, max_iter=0))
    source = torch.zeros(grid.num_cells, dtype=grid.dtype, requires_grad=True)
    eq = make_equation(grid, source)
    # Primal starts at the exact solution; its nonzero cotangent needs a solve.
    with require_converged_solves([solver]):
        result = solver.solve(eq)
    assert not solver.require_convergence
    solver.atol = 1e9
    with pytest.raises(LinearSolveError, match="Transpose"):
        result.solution.sum().backward()
    assert source.grad is None


@pytest.mark.parametrize(
    "method", [SolverType.CG, SolverType.BiCGSTAB, SolverType.PyAMG]
)
def test_successful_strict_solve_and_backward(
    small_axis_projected_grid: AxisProjectedGrid, method: SolverType
) -> None:
    grid = small_axis_projected_grid
    solver = create_solver(SolverConfig(method=method, tolerance=1e-10))
    source = torch.ones(grid.num_cells, dtype=grid.dtype, requires_grad=True)
    eq = make_equation(grid, source)
    with require_converged_solves([solver]):
        with require_converged_solves([solver]):
            result = solver.solve(eq)
        assert solver.require_convergence
    result.solution.sum().backward()
    torch.testing.assert_close(source.grad, torch.full_like(source, 0.5))
    assert not solver.require_convergence
