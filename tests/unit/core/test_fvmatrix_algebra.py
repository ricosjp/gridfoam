"""Unit tests for ``FvMatrix`` linear-algebra operators."""

from __future__ import annotations

import torch

from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.meta.enums import FieldRole


def test_fvmatrix_add_sub_neg(small_axis_projected_grid: AxisProjectedGrid):
    # Element-wise add, subtract, and unary negation must act on all
    # coefficient blocks (diag, upper, lower, source) independently.
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


def test_fvmatrix_A_and_H_operators(
    small_axis_projected_grid: AxisProjectedGrid,
):
    # ``A()`` returns the diagonal scaled by inverse cell volume; ``H(x)``
    # must produce a source-like vector with the same shape as ``x``.
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
