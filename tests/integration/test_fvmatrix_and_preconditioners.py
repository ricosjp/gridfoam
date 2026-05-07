"""Integration tests for FvMatrix algebra and preconditioners on a real grid."""

from __future__ import annotations

import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.meta.enums import FieldRole, PreconditionerType
from gridfoam.solvers.preconditioners import create_preconditioner


def test_fvmatrix_add_sub_neg(small_axis_projected_grid: AxisProjectedGrid):
    grid = small_axis_projected_grid
    p = CellField(grid, "p_alg", role=FieldRole.LOCAL, num_components=1)
    a = FvMatrix(p)
    b = FvMatrix(p)
    torch.manual_seed(0)
    a.diag = torch.randn_like(a.diag)
    a.upper = torch.randn_like(a.upper)
    a.lower = torch.randn_like(a.lower)
    a.source = torch.randn_like(a.source)
    b.diag = torch.randn_like(b.diag)
    b.upper = torch.randn_like(b.upper)
    b.lower = torch.randn_like(b.lower)
    b.source = torch.randn_like(b.source)

    s = a + b
    assert torch.allclose(s.diag, a.diag + b.diag)
    d = a - b
    assert torch.allclose(d.diag, a.diag - b.diag)
    n = -a
    assert torch.allclose(n.diag, -a.diag)


def test_jacobi_preconditioner_scales_by_inverse_diag(
    small_axis_projected_grid: AxisProjectedGrid,
):
    grid = small_axis_projected_grid
    p = CellField(grid, "p_jac", role=FieldRole.LOCAL, num_components=1)
    mat = FvMatrix(p)
    mat.diag = torch.ones_like(mat.diag) * 4.0
    pre = create_preconditioner(PreconditionerType.JACOBI, mat)
    r = torch.ones((grid.num_cells, 1), dtype=grid.dtype, device=grid.device)
    z = pre.apply(r)
    assert torch.allclose(z, r * 0.25)


def test_none_preconditioner_is_identity(
    small_axis_projected_grid: AxisProjectedGrid,
):
    grid = small_axis_projected_grid
    p = CellField(grid, "p_none", role=FieldRole.LOCAL, num_components=1)
    mat = FvMatrix(p)
    pre = create_preconditioner(PreconditionerType.NONE, mat)
    r = torch.randn((grid.num_cells, 1), dtype=grid.dtype, device=grid.device)
    assert torch.allclose(pre.apply(r), r)


def test_fvmatrix_A_and_H_operators(
    small_axis_projected_grid: AxisProjectedGrid,
):
    grid = small_axis_projected_grid
    p = CellField(grid, "p_ah", role=FieldRole.LOCAL, num_components=1)
    mat = FvMatrix(p)
    mat.diag.fill_(3.0)
    mat.source.fill_(1.0)
    x = torch.ones((grid.num_cells, 1), dtype=grid.dtype, device=grid.device)
    Aop = mat.A()
    assert torch.allclose(Aop, mat.diag / grid.cell_volumes)
    H = mat.H(x)
    assert H.shape == x.shape


def test_create_preconditioner_rejects_unknown_type(
    small_axis_projected_grid: AxisProjectedGrid,
):
    grid = small_axis_projected_grid
    p = CellField(grid, "p_bad", role=FieldRole.LOCAL, num_components=1)
    mat = FvMatrix(p)
    with pytest.raises(BeartypeCallHintParamViolation):
        create_preconditioner("not-a-precon", mat)
