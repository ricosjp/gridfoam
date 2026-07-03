"""Unit tests for linear-solver preconditioner factories."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.meta.enums import FieldRole, PreconditionerType
from gridfoam.solvers.preconditioners import create_preconditioner


def test_jacobi_preconditioner_scales_by_inverse_diag(
    small_axis_projected_grid: AxisProjectedGrid,
):
    # Jacobi preconditioning must multiply the residual by 1/diag.
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
    # NONE preconditioner must return the residual unchanged.
    grid = small_axis_projected_grid
    p = CellField(grid, "p_none", role=FieldRole.LOCAL, num_components=1)
    mat = FvMatrix(p)
    pre = create_preconditioner(PreconditionerType.NONE, mat)
    r = torch.randn((grid.num_cells, 1), dtype=grid.dtype, device=grid.device)
    assert torch.allclose(pre.apply(r), r)


def test_create_preconditioner_rejects_unknown_type(
    small_axis_projected_grid: AxisProjectedGrid,
):
    # Factory must reject values outside ``PreconditionerType`` at runtime.
    grid = small_axis_projected_grid
    p = CellField(grid, "p_bad", role=FieldRole.LOCAL, num_components=1)
    mat = FvMatrix(p)
    with pytest.raises(BeartypeCallHintParamViolation):
        create_preconditioner(
            cast(PreconditionerType, cast(object, "not-a-precon")),
            mat,
        )
