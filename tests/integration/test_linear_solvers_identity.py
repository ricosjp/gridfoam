"""Solve identity-like systems with CG, BiCGSTAB, and PyAMG on a small mesh."""

from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.core.equation import Equation, equation
from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.meta.config import SolverConfig
from gridfoam.meta.enums import (
    FieldRole,
    NormType,
    PreconditionerType,
    SolverType,
)
from gridfoam.solvers.factory import create_solver


def _identity_linear_system(
    grid: IGridBase, *, name: str, component_shape: tuple[int, ...]
) -> tuple[Equation, Float[torch.Tensor, " C *component_shape"]]:
    """Build ``I x = b`` as an ``Equation`` with random right-hand side."""
    p = CellField(
        grid, name, role=FieldRole.LOCAL, component_shape=component_shape
    )
    fv_matrix = FvMatrix(p)
    fv_matrix.diag.fill_(1.0)
    fv_matrix.upper.zero_()
    fv_matrix.lower.zero_()
    torch.manual_seed(42)
    b = torch.randn(
        (grid.num_cells, *component_shape), dtype=grid.dtype, device=grid.device
    )
    fv_matrix.source = b.clone()
    p.data.zero_()
    eq = equation(p, fv_matrix)
    return eq, b


def test_cg_solves_identity(small_axis_projected_grid: AxisProjectedGrid):
    # CG without preconditioning must solve a scalar identity system.
    grid = small_axis_projected_grid
    eq, b = _identity_linear_system(grid, name="p_cg", component_shape=())
    cfg = SolverConfig(
        method=SolverType.CG,
        preconditioner=PreconditionerType.NONE,
        tolerance=1e-10,
        rel_tolerance=0.0,
        max_iter=50,
        norm_type=NormType.L_2,
    )
    solver = create_solver(cfg)
    result = solver.solve(eq)
    assert torch.allclose(result.solution, b, atol=1e-6, rtol=1e-6)


def test_bicgstab_solves_identity(small_axis_projected_grid: AxisProjectedGrid):
    # BiCGSTAB must solve a 3-component identity system.
    grid = small_axis_projected_grid
    eq, b = _identity_linear_system(grid, name="p_bicg", component_shape=(3,))
    cfg = SolverConfig(
        method=SolverType.BiCGSTAB,
        preconditioner=PreconditionerType.NONE,
        tolerance=1e-10,
        rel_tolerance=0.0,
        max_iter=100,
        norm_type=NormType.L_2,
        max_restart=5,
    )
    solver = create_solver(cfg)
    result = solver.solve(eq)
    assert torch.allclose(result.solution, b, atol=1e-5, rtol=1e-5)


def test_pyamg_solves_identity(small_axis_projected_grid: AxisProjectedGrid):
    # PyAMG reports independent component statistics through the public API.
    grid = small_axis_projected_grid
    eq, b = _identity_linear_system(grid, name="p_amg", component_shape=(3,))
    cfg = SolverConfig(
        method=SolverType.PyAMG,
        tolerance=1e-12,
        rel_tolerance=1e-10,
        max_iter=200,
    )
    solver = create_solver(cfg)
    result = solver.solve(eq)
    assert torch.allclose(result.solution, b, atol=1e-5, rtol=1e-5)
    assert len(result.stats) == 3
    for c, stats in enumerate(result.stats):
        assert stats.converged
        assert stats.iterations == 1
        initial = torch.linalg.vector_norm(
            b[:, c], ord=cfg.norm_type.to_norm_order()
        ).item()
        assert abs(stats.initial_residual - initial) < 1e-12
        assert stats.final_residual < max(
            cfg.tolerance, cfg.rel_tolerance * initial
        )


def test_equation_factory_returns_named_container(
    small_axis_projected_grid: AxisProjectedGrid,
):
    # ``equation()`` must wrap field and matrix with the field's name.
    grid = small_axis_projected_grid
    p = CellField(grid, "p_eq", role=FieldRole.LOCAL, component_shape=())
    fv_matrix = FvMatrix(p)
    eq = equation(p, fv_matrix)
    assert eq.name == "p_eq"
    assert eq.target is p
    assert eq.fv_matrix is fv_matrix
