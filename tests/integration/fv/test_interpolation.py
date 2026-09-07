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
from gridfoam.fv.fvc.grad import grad
from gridfoam.fv.fvc.interpolate import interpolate
from gridfoam.fv.kernels.face_geometry import face_geometry
from gridfoam.fv.kernels.face_interpolation import (
    correct_internal_values,
    linear_internal_face_values,
)
from gridfoam.meta.enums import FieldRole


def test_linear_internal_face_values_is_exact_on_uniform_mesh():
    """Linear interpolation is exact for a linear field on a uniform mesh."""
    grid = create_grid(small_gridfoam_config())
    field, gradient = linear_scalar_field(grid)
    geo = face_geometry(grid)

    values = linear_internal_face_values(field)
    expected = grid.face_centers[geo.single_mask] @ gradient + 7.0

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
    geo = face_geometry(grid)
    centroid = (
        geo.w_s[:, None] * grid.cell_centers[geo.owner_s]
        + (1.0 - geo.w_s)[:, None] * grid.cell_centers[geo.neighbour_s]
    )
    offset = grid.face_centers[geo.single_mask] - centroid
    offset_faces = torch.linalg.vector_norm(offset, dim=1) > 0.0
    assert bool(offset_faces.any().item())

    base_values = linear_internal_face_values(field)
    grad_data = gradient.expand(grid.num_cells, -1)
    corrected = correct_internal_values(field, base_values, grad_data)
    expected = grid.face_centers[geo.single_mask] @ gradient + 7.0

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
    geo = face_geometry(grid)

    interpolated = interpolate(field)
    expected = grid.face_centers[geo.single_mask] @ gradient + 7.0

    torch.testing.assert_close(
        interpolated.single_data, expected, atol=1e-12, rtol=1e-12
    )


def test_interpolate_is_exact_on_hanging_faces_of_refined_mesh():
    # On a refined mesh, public linear must recover the analytic linear field
    # on hanging-node faces, where the two-point value alone is only O(h).
    grid = refined_grid()
    field, gradient = linear_scalar_field(grid)
    geo = face_geometry(grid)
    assert geo.num_hanging > 0

    base_values = linear_internal_face_values(field)
    corrected = interpolate(field).single_data
    expected = grid.face_centers[geo.single_idx] @ gradient + 7.0

    hang = geo.hang_idx
    assert (base_values[hang] - expected[hang]).abs().max().item() > 1e-3
    torch.testing.assert_close(corrected, expected, atol=1e-12, rtol=1e-12)


def test_interpolate_extrapolates_boundary_faces_without_bcs():
    # A derived field without boundary conditions (e.g. grad(p), HbyA) must
    # receive zero-gradient extrapolated boundary values instead of zeros.
    grid = refined_grid()
    field = CellField(grid, "derived_no_bc", FieldRole.LOCAL, (3,))
    field.data = torch.randn_like(field.data)

    interpolated = interpolate(field)

    expected = field.data[grid.domain_bnd_owner]
    torch.testing.assert_close(
        interpolated.domain_bnd_data, expected, atol=0.0, rtol=0.0
    )


def test_interpolate_of_gradient_field_is_exact_across_hanging_faces():
    # grad(psi) of a linear field is constant and carries no BCs. Its
    # interpolation must stay exact on internal and domain faces; a zero
    # boundary fill would corrupt the skew correction of boundary cells.
    grid = refined_grid()
    field, gradient = linear_scalar_field(grid)
    grad_field = grad(field)

    grad_f = interpolate(grad_field)

    expected_single = gradient.expand_as(grad_f.single_data)
    expected_bnd = gradient.expand_as(grad_f.domain_bnd_data)
    torch.testing.assert_close(
        grad_f.single_data, expected_single, atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        grad_f.domain_bnd_data, expected_bnd, atol=1e-12, rtol=1e-12
    )


def test_interpolate_vector_field_preserves_components():
    # Vector interpolation must keep component count and fill each component
    # independently from the cell values.
    grid = create_grid(small_gridfoam_config())
    field = CellField(grid, "U_interp", FieldRole.LOCAL, (3,))
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
