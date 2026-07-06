"""
Integration tests for cell-centre gradient schemes on refined meshes.

Verifies that ``fvc.grad`` reproduces exact gradients for linear fields and
that the LINEAR scheme's hierarchy-interface correction improves accuracy
over uncorrected Green-Gauss assembly.
"""

from __future__ import annotations

import torch
from tests.helpers import (
    interior_mask,
    linear_scalar_field,
    refined_3d_grid,
)

from gridfoam.fv import fvc
from gridfoam.fv.fvc.interpolate import (
    interpolate,
    linear_internal_face_values,
)
from gridfoam.fv.schemes.grad import gauss_assemble
from gridfoam.meta.enums import GradScheme


def test_leastsquare_grad_is_linear_exact_on_refined_internal_cells():
    # LEASTSQUARE must recover a constant gradient on interior cells of a
    # 3-D octree-refined mesh (no hierarchy-interface correction needed).
    grid = refined_3d_grid(GradScheme.LEASTSQUARE)
    field, expected = linear_scalar_field(grid)
    interior = interior_mask(grid)

    grad_field = fvc.grad(field)

    torch.testing.assert_close(
        grad_field.data[interior],
        expected.expand_as(grad_field.data[interior]),
        atol=1e-12,
        rtol=1e-12,
    )


def test_linear_grad_corrects_interface_error_on_refined_mesh():
    # LINEAR grad must be more accurate than uncorrected Green-Gauss on a
    # 3-D refined mesh where face values straddle hierarchy interfaces.
    grid = refined_3d_grid(GradScheme.LINEAR)
    field, expected_vec = linear_scalar_field(grid)
    interior = interior_mask(grid)
    expected = expected_vec.to(grid.device)

    psi_f = interpolate(field)
    uncorrected = gauss_assemble(
        field, psi_f, linear_internal_face_values(field)
    )
    corrected = fvc.grad(field).data

    uncorrected_err = torch.linalg.vector_norm(
        uncorrected[interior] - expected
    ).item()
    corrected_err = torch.linalg.vector_norm(
        corrected[interior] - expected
    ).item()
    assert corrected_err < uncorrected_err
