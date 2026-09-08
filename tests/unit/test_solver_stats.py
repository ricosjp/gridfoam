"""
CG and BiCGSTAB report component-wise residuals, iterations, and
convergence decisions.
"""

from __future__ import annotations

import pytest
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
from gridfoam.solvers.factory import create_solver


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
    """
    Scalar CG returns one converged statistic with nonnegative residuals
    and iterations.
    """
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
    """
    Vector CG retains the solution shape and returns one converged
    statistic per component.
    """
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


@pytest.mark.parametrize(
    ("atol", "rtol", "expected_x", "iterations", "converged"),
    [
        (2.0, 0.0, (0.0, 0.0), 0, True),
        (0.6, 0.0, (2 / 3, 2 / 3), 1, True),
        (0.2, 0.0, (13 / 15, 7 / 15), 1, True),
        (1e-12, 0.5, (2 / 3, 2 / 3), 1, True),
        (1e-12, 0.2, (13 / 15, 7 / 15), 1, True),
        (1e-12, 0.0, (13 / 15, 7 / 15), 1, False),
    ],
)
def test_bicgstab_convergence_stages(
    small_axis_projected_grid: AxisProjectedGrid,
    atol: float,
    rtol: float,
    expected_x: tuple[float, float],
    iterations: int,
    converged: bool,
) -> None:
    """
    BiCGSTAB reports the solution and true residual at initial, s, r, and
    budget exits.
    """
    grid = small_axis_projected_grid
    eq = _identity_equation(grid, name="p_bicg_stats")
    mat = eq.fv_matrix
    mat.diag[1] = 2.0
    mat.source.zero_()
    mat.source[:2] = 1.0
    solver = create_solver(
        SolverConfig(
            method=SolverType.BiCGSTAB,
            preconditioner=PreconditionerType.NONE,
            tolerance=atol,
            rel_tolerance=rtol,
            max_iter=1,
            norm_type=NormType.L_2,
        )
    )
    result = solver.solve(eq)
    expected = torch.zeros_like(mat.source)
    expected[:2] = expected.new_tensor(expected_x)
    torch.testing.assert_close(result.solution, expected)
    stats = result.stats[0]
    assert stats.iterations == iterations
    assert stats.converged is converged
    assert stats.initial_residual == pytest.approx(2**0.5)
    assert stats.final_residual == pytest.approx(
        torch.linalg.vector_norm(mat.source - mat.multiply(expected)).item()
    )
