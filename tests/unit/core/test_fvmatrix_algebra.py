"""
Matrix diagonals obey arithmetic; source replacement preserves gradients
and storage isolation.
"""

from __future__ import annotations

import pytest
import torch

from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.meta.enums import FieldRole


def test_fvmatrix_diagonal_obeys_add_sub_neg(
    small_axis_projected_grid: AxisProjectedGrid,
) -> None:
    """
    Addition, subtraction, and negation apply elementwise to the matrix
    diagonal.
    """
    grid = small_axis_projected_grid
    p = CellField(grid, "p_alg", role=FieldRole.LOCAL, component_shape=())
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
) -> None:
    """
    ``A()`` returns the diagonal scaled by inverse cell volume; ``H(x)``
    must produce a source-like vector with the same shape as ``x``.
    """
    grid = small_axis_projected_grid
    p = CellField(grid, "p_ah", role=FieldRole.LOCAL, component_shape=())
    mat = FvMatrix(p)
    mat.diag.fill_(3.0)
    mat.source.fill_(1.0)
    x = torch.ones((grid.num_cells,), dtype=grid.dtype, device=grid.device)
    Aop = mat.A()
    assert torch.allclose(Aop, mat.diag / grid.cell_volumes)
    H = mat.H(x)
    assert H.shape == x.shape


@pytest.mark.parametrize("shape", [(), (3,), (3, 3)])
def test_predictor_source_keeps_base_matrix_independent(
    small_axis_projected_grid: AxisProjectedGrid, shape: tuple[int, ...]
) -> None:
    """
    Source replacement preserves gradients and isolates every mutable
    matrix block.
    """
    grid = small_axis_projected_grid
    field = CellField(grid, "predictor_field", FieldRole.LOCAL, shape)
    base = FvMatrix(field)
    base.diag.fill_(2.0)
    base.upper.fill_(-0.2)
    base.lower.fill_(-0.3)
    base.source = torch.ones_like(base.source, requires_grad=True)
    base.face_flux_correction = torch.ones(
        (grid.num_internal_faces, *shape), dtype=grid.dtype, device=grid.device
    )
    force = torch.full_like(base.source, 0.5, requires_grad=True)
    predictor = base.with_source(base.source + force)

    # Adding a source affects H but not A, and does not contaminate base.H.
    x = torch.ones_like(base.source)
    volumes = grid.cell_volumes.reshape(-1, *(1 for _ in shape))
    torch.testing.assert_close(predictor.A(), base.A())
    torch.testing.assert_close(predictor.H(x) - base.H(x), force / volumes)
    source_grad, force_grad = torch.autograd.grad(
        predictor.source.sum(), (base.source, force)
    )
    torch.testing.assert_close(source_grad, torch.ones_like(base.source))
    torch.testing.assert_close(force_grad, torch.ones_like(force))

    # Later solver/constraint operations must not mutate the base matrix,
    # including its explicit face-flux correction.
    for name in ("diag", "upper", "lower", "source", "face_flux_correction"):
        base_tensor = getattr(base, name)
        predictor_tensor = getattr(predictor, name)
        before = base_tensor.clone()
        with torch.no_grad():
            predictor_tensor.add_(1.0)
        torch.testing.assert_close(base_tensor, before)


def test_with_source_preserves_absent_face_correction(
    small_axis_projected_grid: AxisProjectedGrid,
) -> None:
    """
    Replacing a source does not create an absent explicit face-flux
    correction.
    """
    field = CellField(
        small_axis_projected_grid,
        "predictor_no_correction",
        FieldRole.LOCAL,
        (),
    )
    base = FvMatrix(field)
    assert base.with_source(base.source).face_flux_correction is None
