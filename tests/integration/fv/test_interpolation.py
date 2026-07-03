"""
Integration tests for face interpolation on uniform and refined meshes.

Covers linear internal face reconstruction, corrected-linear face values at
octree interfaces, and the ``interpolate`` entry point.
"""

from __future__ import annotations

import torch
from tests.conftest import small_gridfoam_config

from gridfoam.core.grid.factory import create_grid
from gridfoam.fv.fvc.interpolate import (
    interpolate,
    linear_internal_face_values,
    single_face_linear_weights,
    single_internal_mask,
)
from gridfoam.fv.schemes.grad import corrected_linear_internal_face_values
from tests.helpers import linear_scalar_field, refined_grid


def test_linear_internal_face_values_is_exact_on_uniform_mesh():
    # On a uniform mesh, linear interpolation from cell values must match
    # the analytic linear field evaluated at face centres.
    grid = create_grid(small_gridfoam_config())
    field, gradient = linear_scalar_field(grid)
    single_mask = single_internal_mask(grid)

    linear = linear_internal_face_values(field)
    expected = (grid.face_centers[single_mask] @ gradient + 7.0).reshape(-1, 1)

    torch.testing.assert_close(linear, expected, atol=1e-12, rtol=1e-12)


def test_corrected_linear_face_values_match_linear_on_uniform_mesh():
    # Corrected-linear reconstruction collapses to plain linear when the
    # face offset from the centroid segment is zero (uniform mesh).
    grid = create_grid(small_gridfoam_config())
    field, gradient = linear_scalar_field(grid)
    grad_data = gradient.expand(grid.num_cells, -1)

    linear = linear_internal_face_values(field)
    corrected = corrected_linear_internal_face_values(field, grad_data)

    torch.testing.assert_close(corrected, linear, atol=1e-12, rtol=1e-12)


def test_corrected_linear_face_values_improve_where_face_offset_is_nonzero():
    # On a refined mesh, corrected-linear must be closer to the exact face
    # value than uncorrected linear wherever face centres are offset from
    # the owner/neighbour centroid segment.
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

    linear = linear_internal_face_values(field)
    grad_data = gradient.expand(grid.num_cells, -1)
    corrected = corrected_linear_internal_face_values(field, grad_data)
    expected = (grid.face_centers[single_mask] @ gradient + 7.0).reshape(-1, 1)

    linear_err = (
        (linear[offset_faces] - expected[offset_faces]).abs().max().item()
    )
    corrected_err = (
        (corrected[offset_faces] - expected[offset_faces]).abs().max().item()
    )
    assert corrected_err < linear_err


def test_interpolate_uses_linear_internal_face_values():
    # ``interpolate`` must delegate to linear internal face values on a
    # uniform mesh (single-face data should match exactly).
    grid = create_grid(small_gridfoam_config())
    field, _ = linear_scalar_field(grid)

    interpolated = interpolate(field)
    linear = linear_internal_face_values(field)
    torch.testing.assert_close(
        interpolated.single_data, linear, atol=1e-12, rtol=1e-12
    )
