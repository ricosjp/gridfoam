"""Gradient checks for CG / BiCGSTAB / PyAMG implicit adjoints."""

from __future__ import annotations

import pytest
import torch
from jaxtyping import Float

from gridfoam.core.equation import Equation, equation
from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import GridBase
from gridfoam.meta.config import SolverConfig
from gridfoam.meta.enums import (
    FieldRole,
    NormType,
    PreconditionerType,
    SolverType,
)
from gridfoam.solvers.factory import create_solver


def _diagonally_dominant_system(
    grid: GridBase,
    *,
    name: str,
    symmetric: bool,
    seed: int = 0,
) -> tuple[
    Equation,
    Float[torch.Tensor, " C"],
    Float[torch.Tensor, " F"],
    Float[torch.Tensor, " F"],
    Float[torch.Tensor, " C"],
]:
    """
    Build a diagonally-dominant scalar LDU system in float64.

    When ``symmetric`` is True, ``upper == lower`` so CG and PyAMG apply.
    """
    dtype = torch.float64
    device = grid.device
    p = CellField(grid, name, role=FieldRole.LOCAL, component_shape=())
    # Force float64 storage even if the grid fixture uses another dtype.
    p.data = torch.zeros((grid.num_cells,), dtype=dtype, device=device)
    fv_matrix = FvMatrix(p)
    n_faces = grid.num_internal_faces
    n_cells = grid.num_cells

    torch.manual_seed(seed)
    upper = 0.1 * torch.rand(n_faces, dtype=dtype, device=device)
    lower = (
        upper.clone()
        if symmetric
        else (0.05 * torch.rand(n_faces, dtype=dtype, device=device))
    )

    # Accumulate off-diagonal magnitude per cell for diagonal dominance.
    off_diag = torch.zeros(n_cells, dtype=dtype, device=device)
    off_diag.index_add_(0, grid.owner, upper.abs())
    off_diag.index_add_(0, grid.neighbour, lower.abs())
    diag = off_diag + 1.0

    source = torch.randn(n_cells, dtype=dtype, device=device)

    fv_matrix.diag = diag
    fv_matrix.upper = upper
    fv_matrix.lower = lower
    fv_matrix.source = source
    p.data.zero_()

    eq = equation(p, fv_matrix)
    return eq, diag, upper, lower, source


def _solver_config(method: SolverType) -> SolverConfig:
    return SolverConfig(
        method=method,
        preconditioner=PreconditionerType.JACOBI,
        tolerance=1e-12,
        rel_tolerance=1e-12,
        max_iter=200,
        norm_type=NormType.L_2,
        max_restart=10,
    )


def _grad_wrt_coeffs(
    method: SolverType,
    eq: Equation,
    diag: torch.Tensor,
    upper: torch.Tensor,
    lower: torch.Tensor,
    source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve once and return analytic grads of ``sum(x)`` w.r.t. coeffs."""
    diag = diag.detach().requires_grad_(True)
    upper = upper.detach().requires_grad_(True)
    lower = lower.detach().requires_grad_(True)
    source = source.detach().requires_grad_(True)

    A = eq.fv_matrix
    A.diag = diag
    A.upper = upper
    A.lower = lower
    A.source = source
    eq.target.data.zero_()

    solver = create_solver(_solver_config(method))
    x = solver.solve(eq).solution
    x.sum().backward()

    assert diag.grad is not None
    assert upper.grad is not None
    assert lower.grad is not None
    assert source.grad is not None
    return diag.grad, upper.grad, lower.grad, source.grad


def test_as_transpose_matches_dot(
    small_axis_projected_grid: AxisProjectedGrid,
) -> None:
    """y·(Ax) must equal (Aᵀy)·x for a random non-symmetric LDU matrix."""
    grid = small_axis_projected_grid
    p = CellField(grid, "p_tr", role=FieldRole.LOCAL, component_shape=())
    A = FvMatrix(p)
    torch.manual_seed(1)
    A.diag = torch.randn_like(A.diag)
    A.upper = torch.randn_like(A.upper)
    A.lower = torch.randn_like(A.lower)
    x = torch.randn_like(A.source)
    y = torch.randn_like(A.source)

    A_T = A.as_transpose()
    lhs = torch.sum(y * A.multiply(x))
    rhs = torch.sum(A_T.multiply(y) * x)
    assert torch.allclose(lhs, rhs, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    ("method", "symmetric", "seed"),
    [
        (SolverType.CG, True, 0),
        (SolverType.BiCGSTAB, False, 2),
    ],
)
def test_krylov_adjoint_gradcheck(
    small_axis_projected_grid: AxisProjectedGrid,
    method: SolverType,
    symmetric: bool,
    seed: int,
) -> None:
    """Finite-difference check of Krylov implicit adjoint on LDU coeffs."""
    grid = small_axis_projected_grid
    eq, diag0, upper0, lower0, source0 = _diagonally_dominant_system(
        grid,
        name=f"p_{method.name.lower()}_adj",
        symmetric=symmetric,
        seed=seed,
    )
    solver = create_solver(_solver_config(method))

    def solve_fn(
        diag: torch.Tensor,
        upper: torch.Tensor,
        lower: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        A = eq.fv_matrix
        A.diag = diag
        A.upper = upper
        A.lower = lower
        A.source = source
        eq.target.data.zero_()
        return solver.solve(eq).solution

    inputs = (
        diag0.detach().requires_grad_(True),
        upper0.detach().requires_grad_(True),
        lower0.detach().requires_grad_(True),
        source0.detach().requires_grad_(True),
    )
    assert torch.autograd.gradcheck(
        solve_fn,
        inputs,
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
    )


def test_cg_adjoint_matches_pyamg(
    small_axis_projected_grid: AxisProjectedGrid,
) -> None:
    """
    On a symmetric diagonally-dominant system, CG and PyAMG implicit
    adjoints must agree on coefficient gradients.
    """
    grid = small_axis_projected_grid
    eq, diag, upper, lower, source = _diagonally_dominant_system(
        grid, name="p_cmp", symmetric=True, seed=3
    )

    g_cg = _grad_wrt_coeffs(SolverType.CG, eq, diag, upper, lower, source)
    g_amg = _grad_wrt_coeffs(SolverType.PyAMG, eq, diag, upper, lower, source)

    for a, b in zip(g_cg, g_amg, strict=True):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)
