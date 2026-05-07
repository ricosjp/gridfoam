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
    grid: IGridBase, *, name: str, k: int
) -> tuple[Equation, Float[torch.Tensor, " C K"]]:
    p = CellField(grid, name, role=FieldRole.LOCAL, num_components=k)
    lhs = FvMatrix(p)
    lhs.diag.fill_(1.0)
    lhs.upper.zero_()
    lhs.lower.zero_()
    torch.manual_seed(42)
    b = torch.randn(grid.num_cells, k, dtype=grid.dtype, device=grid.device)
    lhs.source = b.clone()
    p.data.zero_()
    eq = equation("p", p, lhs)
    return eq, b


def test_cg_solves_identity(small_axis_projected_grid: AxisProjectedGrid):
    grid = small_axis_projected_grid
    eq, b = _identity_linear_system(grid, name="p_cg", k=1)
    cfg = SolverConfig(
        method=SolverType.CG,
        preconditioner=PreconditionerType.NONE,
        tolerance=1e-10,
        rel_tolerance=0.0,
        max_iter=50,
        norm_type=NormType.L_2,
    )
    solver = create_solver(cfg)
    x = solver.solve(eq)
    assert torch.allclose(x, b, atol=1e-6, rtol=1e-6)


def test_bicgstab_solves_identity(small_axis_projected_grid: AxisProjectedGrid):
    grid = small_axis_projected_grid
    eq, b = _identity_linear_system(grid, name="p_bicg", k=2)
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
    x = solver.solve(eq)
    assert torch.allclose(x, b, atol=1e-5, rtol=1e-5)


def test_pyamg_solves_identity(small_axis_projected_grid: AxisProjectedGrid):
    grid = small_axis_projected_grid
    eq, b = _identity_linear_system(grid, name="p_amg", k=1)
    cfg = SolverConfig(
        method=SolverType.PyAMG,
        tolerance=1e-12,
        rel_tolerance=1e-10,
        max_iter=200,
    )
    solver = create_solver(cfg)
    x = solver.solve(eq)
    assert torch.allclose(x, b, atol=1e-5, rtol=1e-5)


def test_equation_factory_returns_named_container(
    small_axis_projected_grid: AxisProjectedGrid,
):
    grid = small_axis_projected_grid
    p = CellField(grid, "p_eq", role=FieldRole.LOCAL, num_components=1)
    lhs = FvMatrix(p)
    eq = equation("pressure", p, lhs)
    assert eq.name == "pressure"
    assert eq.target is p
    assert eq.lhs is lhs
