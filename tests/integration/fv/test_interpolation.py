"""
Integration tests for face interpolation on uniform and refined meshes.

Covers hierarchy-aware linear interpolation and boundary storage.
"""

from __future__ import annotations

import torch
from tests.conftest import small_gridfoam_config
from tests.helpers import linear_scalar_field, refined_grid

from gridfoam.core.field import CellField
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv.fvc.interpolate import interpolate
from gridfoam.fv.kernels.face_interpolation import (
    correct_internal_values,
    linear_internal_face_values,
    single_face_linear_weights,
    single_internal_mask,
)
from gridfoam.meta.enums import FieldRole


def test_linear_internal_face_values_is_exact_on_uniform_mesh():
    """Linear interpolation is exact for a linear field on a uniform mesh."""
    grid = create_grid(small_gridfoam_config())
    field, gradient = linear_scalar_field(grid)
    single_mask = single_internal_mask(grid)

    values = linear_internal_face_values(field)
    expected = (grid.face_centers[single_mask] @ gradient + 7.0).reshape(-1, 1)

    torch.testing.assert_close(values, expected, atol=1e-12, rtol=1e-12)


def test_correct_internal_values_match_base_values_on_uniform_mesh():
    """Face-offset correction vanishes on a uniform mesh."""
    grid = create_grid(small_gridfoam_config())
    field, gradient = linear_scalar_field(grid)
    grad_data = gradient.expand(grid.num_cells, -1)

    base_values = linear_internal_face_values(field)
    corrected = correct_internal_values(field, base_values, grad_data)

    torch.testing.assert_close(corrected, base_values, atol=1e-12, rtol=1e-12)


def test_correct_internal_values_improve_on_offset_faces():
    """Face-offset correction improves linear values on a refined mesh."""
    grid = refined_grid()
    field, gradient = linear_scalar_field(grid)
    single_mask = single_internal_mask(grid)
    owner, neighbour, w = single_face_linear_weights(grid, single_mask)
    centroid = (
        w * grid.cell_centers[owner] + (1.0 - w) * grid.cell_centers[neighbour]
    )
    offset = grid.face_centers[single_mask] - centroid
    offset_faces = torch.linalg.vector_norm(offset, dim=1) > 0.0
    assert bool(offset_faces.any().item())

    base_values = linear_internal_face_values(field)
    grad_data = gradient.expand(grid.num_cells, -1)
    corrected = correct_internal_values(field, base_values, grad_data)
    expected = (grid.face_centers[single_mask] @ gradient + 7.0).reshape(-1, 1)

    base_error = (
        (base_values[offset_faces] - expected[offset_faces]).abs().max().item()
    )
    corrected_error = (
        (corrected[offset_faces] - expected[offset_faces]).abs().max().item()
    )
    assert corrected_error < base_error


def test_interpolate_is_exact_on_uniform_mesh():
    # On a uniform mesh, hierarchy-aware linear must match the analytic
    # linear field at face centres (the offset correction vanishes).
    grid = create_grid(small_gridfoam_config())
    field, gradient = linear_scalar_field(grid)
    single_mask = single_internal_mask(grid)

    interpolated = interpolate(field)
    expected = (grid.face_centers[single_mask] @ gradient + 7.0).reshape(-1, 1)

    torch.testing.assert_close(
        interpolated.single_data, expected, atol=1e-12, rtol=1e-12
    )


def test_interpolate_improves_where_face_offset_is_nonzero():
    # On a refined mesh, public linear must beat the analytic two-point
    # interpolation wherever face centres are offset from the owner/neighbour
    # centroid segment.
    grid = refined_grid()
    field, gradient = linear_scalar_field(grid)
    single_mask = single_internal_mask(grid)
    owner, neighbour, w = single_face_linear_weights(grid, single_mask)
    centroid = (
        w * grid.cell_centers[owner] + (1.0 - w) * grid.cell_centers[neighbour]
    )
    offset = grid.face_centers[single_mask] - centroid
    offset_faces = torch.linalg.vector_norm(offset, dim=1) > 0.0
    if not bool(offset_faces.any().item()):
        return

    base_values = linear_internal_face_values(field)
    corrected = interpolate(field).single_data
    expected = (grid.face_centers[single_mask] @ gradient + 7.0).reshape(-1, 1)

    base_error = (
        (base_values[offset_faces] - expected[offset_faces]).abs().max().item()
    )
    corrected_err = (
        (corrected[offset_faces] - expected[offset_faces]).abs().max().item()
    )
    assert corrected_err < base_error


def test_interpolate_vector_field_preserves_components():
    # Vector interpolation must keep component count and fill each component
    # independently from the cell values.
    grid = create_grid(small_gridfoam_config())
    field = CellField(grid, "U_interp", FieldRole.LOCAL, 3)
    field.data = grid.cell_centers.clone()

    interpolated = interpolate(field)
    assert interpolated.num_components == 3
    assert interpolated.single_data.shape[1] == 3
    torch.testing.assert_close(
        interpolated.single_data,
        grid.face_centers[interpolated.single_mask],
        atol=1e-12,
        rtol=1e-12,
    )


def test_interpolate_writes_domain_boundary_storage():
    # Domain-boundary slots must be filled from BC evaluation and remain
    # separate from single-sided internal data.
    grid = create_grid(small_gridfoam_config())
    field, _ = linear_scalar_field(grid)
    interpolated = interpolate(field)

    assert interpolated.domain_bnd_data.shape[0] == grid.num_domain_bnd_faces
    assert torch.all(torch.isfinite(interpolated.domain_bnd_data))
    assert interpolated.single_data.shape[0] == int(
        interpolated.single_mask.sum().item()
    )
