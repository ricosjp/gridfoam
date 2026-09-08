"""
Cell gradients preserve linear exactness and physical/derivative axis order.

Accuracy
    Least-squares and corrected Green-Gauss recover linear gradients on
    refined meshes. Constant fields with zero-gradient BCs have zero gradient.

Configuration and layout
    Omitted configuration remains linear-exact at boundaries and hanging
    cells. A nonsymmetric vector Jacobian detects transposed output axes.
"""

from __future__ import annotations

import torch
from tests.helpers import (
    interior_mask,
    linear_scalar_field,
    refined_3d_grid,
    refined_grid,
)

from gridfoam.boundaries.basic.neumann import NeumannBC
from gridfoam.core.field import CellField
from gridfoam.fv import fvc
from gridfoam.fv.kernels.face_interpolation import linear_internal_face_values
from gridfoam.fv.kernels.gauss_gradient import assemble_gauss_gradient
from gridfoam.meta.enums import DomainBoundaryPatch, FieldRole, GradScheme


def test_leastsquare_grad_is_linear_exact_on_refined_internal_cells() -> None:
    """
    LEASTSQUARE must recover a constant gradient on interior cells of a 3-D
    octree-refined mesh.
    """
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


def test_linear_grad_is_linear_exact_on_refined_mesh() -> None:
    """
    LINEAR (Green-Gauss) must recover a constant gradient everywhere;
    Green-Gauss from uncorrected two-point faces is inconsistent on hanging
    cells.
    """
    grid = refined_3d_grid(GradScheme.LINEAR)
    field, expected_vec = linear_scalar_field(grid)
    expected = expected_vec.to(grid.device)

    uncorrected_faces = fvc.interpolate(field)
    uncorrected_faces.single_data = linear_internal_face_values(field)
    uncorrected = assemble_gauss_gradient(grid, uncorrected_faces)
    corrected = fvc.grad(field).data

    assert (uncorrected - expected).abs().max().item() > 1e-2
    torch.testing.assert_close(
        corrected, expected.expand_as(corrected), atol=1e-12, rtol=1e-12
    )


def test_default_grad_is_linear_exact_at_boundary_and_hanging_cells() -> None:
    """Omitted grad configuration remains linear-exact in every cell."""
    grid = refined_3d_grid()
    assert grid.sim_config.fvSchemes.gradSchemes is None
    field, expected = linear_scalar_field(grid)

    grad_field = fvc.grad(field)

    torch.testing.assert_close(
        grad_field.data,
        expected.expand_as(grad_field.data),
        atol=1e-12,
        rtol=1e-12,
    )


def test_grad_vector_output_layout() -> None:
    """A nonsymmetric Jacobian detects transposed physical/derivative axes."""
    grid = refined_3d_grid(GradScheme.LEASTSQUARE)
    field = CellField(grid, "U_grad", FieldRole.LOCAL, (3,))
    jacobian = torch.tensor(
        [[1.0, 2.0, -3.0], [4.0, -2.0, 1.0], [0.5, 3.0, 2.0]],
        dtype=grid.dtype,
        device=grid.device,
    )
    field.data = grid.cell_centers @ jacobian.T
    bcs = {}
    for patch in DomainBoundaryPatch:
        direction = patch.to_direction().value
        sign = 2.0 * (direction % 2) - 1.0
        bcs[patch] = NeumannBC(sign * jacobian[:, direction // 2])
    field.add_boundary_conditions(bcs)
    grad_field = fvc.grad(field)
    assert grad_field.component_shape == (3, 3)
    assert grad_field.num_components == 9
    assert grad_field.data.shape == (grid.num_cells, 3, 3)
    torch.testing.assert_close(
        grad_field.data,
        jacobian.expand_as(grad_field.data),
        atol=1e-12,
        rtol=1e-12,
    )


def test_constant_field_has_zero_gradient() -> None:
    """Constant field with zero-gradient boundaries → vanishing gradient."""
    grid = refined_grid(grad_scheme=GradScheme.LINEAR)
    field = CellField(grid, "const", FieldRole.LOCAL, ())
    field.data[:] = 3.0
    zero = torch.zeros((), dtype=grid.dtype, device=grid.device)
    field.add_boundary_conditions(
        {patch: NeumannBC(zero) for patch in DomainBoundaryPatch}
    )

    grad_field = fvc.grad(field)
    torch.testing.assert_close(
        grad_field.data,
        torch.zeros_like(grad_field.data),
        atol=1e-12,
        rtol=1e-12,
    )
