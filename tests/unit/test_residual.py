"""Unit tests for residual helpers."""

from __future__ import annotations

import torch

from gridfoam.algorithms.utils.residual import (
    continuity_residual,
    field_initial_residual,
    residual_satisfied,
)
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv import fvm
from gridfoam.meta.enums import FieldRole


def test_field_initial_residual_zero_for_exact_solution(
    small_axis_projected_grid: AxisProjectedGrid,
):
    # Residual must be zero when the field satisfies the Laplacian system.
    grid = small_axis_projected_grid
    p = CellField(grid, "p_res", role=FieldRole.LOCAL, component_shape=())
    mat = fvm.laplacian(1.0, p)
    p.data.fill_(0.0)
    mat.source.fill_(0.0)
    assert field_initial_residual(mat, p) == 0.0


def test_continuity_residual_matches_div_norm(
    small_axis_projected_grid: AxisProjectedGrid,
):
    # Continuity residual must be a non-negative scalar from div(phi).
    grid = small_axis_projected_grid
    phi = FaceField(grid, "phi_res", role=FieldRole.LOCAL, component_shape=())
    phi.single_data = torch.randn_like(phi.single_data)
    residual = continuity_residual(phi)
    assert residual >= 0.0


def test_residual_satisfied_absolute_or_relative():
    # OpenFOAM residualControl: absTol or relTol, not both.
    assert residual_satisfied(1e-9, 1e-6, 0.0, 1.0)
    assert not residual_satisfied(1e-3, 1e-6, 0.0, 1.0)
    # Absolute miss, relative hit.
    assert residual_satisfied(5e-5, 1e-6, 0.1, 1e-3)
    # Absolute hit, relative miss.
    assert residual_satisfied(1e-7, 1e-6, 0.1, 1e-8)
    assert not residual_satisfied(2e-4, 1e-6, 0.1, 1e-3)


def test_fvmatrix_residual_nonzero_for_mismatch(
    small_axis_projected_grid: AxisProjectedGrid,
):
    # A field that does not satisfy Ax=b must yield a positive residual.
    grid = small_axis_projected_grid
    p = CellField(grid, "p_bad", role=FieldRole.LOCAL, component_shape=())
    mat = FvMatrix(p)
    mat.diag.fill_(2.0)
    mat.source.fill_(1.0)
    p.data.fill_(1.0)
    assert field_initial_residual(mat, p) > 0.0
