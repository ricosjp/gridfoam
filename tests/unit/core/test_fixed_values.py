"""Symmetric elimination agrees with a dense constrained solve and adjoint."""

import pytest
import torch

from gridfoam.core.equation import equation
from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.meta.config import SolverConfig
from gridfoam.meta.enums import FieldRole, SolverType
from gridfoam.solvers.factory import create_solver


@pytest.mark.parametrize("shape", [(), (3,), (3, 3)])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_fixed_values_match_dense_system_and_gradients(
    small_axis_projected_grid: AxisProjectedGrid,
    shape: tuple[int, ...],
    sign: float,
):
    grid = small_axis_projected_grid
    field = CellField(grid, "fixed_test", FieldRole.LOCAL, shape)
    matrix = FvMatrix(field)
    matrix.diag = torch.full_like(matrix.diag, 10.0 * sign)
    matrix.upper = torch.full_like(matrix.upper, -sign)
    matrix.lower = matrix.upper.clone()
    generator = torch.Generator().manual_seed(3)
    matrix.source = torch.randn(
        matrix.source.shape, dtype=grid.dtype, generator=generator
    )
    source_before = matrix.source.clone()
    cells = torch.tensor([0, 1, grid.num_cells - 1])
    values = torch.randn(
        (cells.numel(), *shape),
        dtype=grid.dtype,
        generator=generator,
        requires_grad=True,
    )
    constrained = matrix.with_fixed_values(cells, values)
    torch.testing.assert_close(constrained.upper, constrained.lower)
    torch.testing.assert_close(matrix.source, source_before)
    solver = create_solver(
        SolverConfig(
            method=SolverType.CG,
            tolerance=1e-12,
            rel_tolerance=0.0,
            max_iter=100,
        )
    )
    result = solver.solve(equation(field, constrained))
    assert all(stat.converged for stat in result.stats)
    torch.testing.assert_close(
        result.solution[cells], values, atol=1e-12, rtol=0
    )

    # Independent reduced dense system: retain only the free unknowns.
    dense = torch.diag(matrix.diag)
    dense.index_put_(
        (grid.owner, grid.neighbour), matrix.upper, accumulate=True
    )
    dense.index_put_(
        (grid.neighbour, grid.owner), matrix.lower, accumulate=True
    )
    free = torch.ones(grid.num_cells, dtype=torch.bool)
    free[cells] = False
    flat_values = values.reshape(cells.numel(), -1)
    rhs = (
        matrix.source.reshape(grid.num_cells, -1)[free]
        - dense[free][:, cells] @ flat_values
    )
    free_solution = torch.linalg.solve(dense[free][:, free], rhs)
    expected = torch.zeros_like(matrix.source).reshape(grid.num_cells, -1)
    expected[free] = free_solution
    expected = expected.index_copy(0, cells, flat_values).reshape_as(
        matrix.source
    )
    torch.testing.assert_close(result.solution, expected, atol=1e-11, rtol=0)
    actual_gradient = torch.autograd.grad(
        result.solution.square().sum(), values
    )[0]
    expected_gradient = torch.autograd.grad(expected.square().sum(), values)[0]
    torch.testing.assert_close(
        actual_gradient, expected_gradient, atol=1e-10, rtol=0
    )
