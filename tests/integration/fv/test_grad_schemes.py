"""
Integration tests for cell-centre gradient schemes on refined meshes.

Public ``fvc.grad`` contracts, hierarchy accuracy, and layout.
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


def test_leastsquare_grad_is_linear_exact_on_refined_internal_cells():
    # LEASTSQUARE must recover a constant gradient on interior cells of a
    # 3-D octree-refined mesh.
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


def test_linear_grad_is_linear_exact_on_refined_mesh():
    # LINEAR (Green-Gauss) must recover a constant gradient everywhere;
    # Green-Gauss from uncorrected two-point faces is inconsistent on
    # hanging cells.
    grid = refined_3d_grid(GradScheme.LINEAR)
    field, expected_vec = linear_scalar_field(grid)
    expected = expected_vec.to(grid.device)

    uncorrected_faces = fvc.interpolate(field)
    uncorrected_faces.single_data = linear_internal_face_values(field)
    uncorrected = assemble_gauss_gradient(grid, uncorrected_faces)[:, 0, :]
    corrected = fvc.grad(field).data

    assert (uncorrected - expected).abs().max().item() > 1e-2
    torch.testing.assert_close(
        corrected, expected.expand_as(corrected), atol=1e-12, rtol=1e-12
    )


def test_default_grad_scheme_is_leastsquare_and_exact_everywhere():
    # Without a ``gradSchemes`` entry the default is least-squares, exact
    # on boundary and hanging cells.
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


def test_grad_vector_output_layout():
    # Vector gradients occupy contiguous [3*c:3*c+3] blocks.
    grid = refined_grid(grad_scheme=GradScheme.LEASTSQUARE)
    field = CellField(grid, "U_grad", FieldRole.LOCAL, 2)
    field.data[:, 0:1] = (
        grid.cell_centers
        @ torch.tensor([1.0, 0.0, 0.0], dtype=grid.dtype, device=grid.device)
    ).reshape(-1, 1)
    field.data[:, 1:2] = (
        grid.cell_centers
        @ torch.tensor([0.0, 2.0, 0.0], dtype=grid.dtype, device=grid.device)
    ).reshape(-1, 1)
    bcs = {}
    for patch in DomainBoundaryPatch:
        direction = patch.to_direction()
        axis = direction.value // 2
        sign = 2.0 * (direction.value % 2) - 1.0
        g0 = 1.0 if axis == 0 else 0.0
        g1 = 2.0 if axis == 1 else 0.0
        bcs[patch] = NeumannBC(
            torch.tensor(
                [sign * g0, sign * g1], dtype=grid.dtype, device=grid.device
            )
        )
    field.add_boundary_conditions(bcs)

    grad_field = fvc.grad(field)
    assert grad_field.num_components == 6
    assert grad_field.data.shape == (grid.num_cells, 6)

    interior = interior_mask(grid)
    torch.testing.assert_close(
        grad_field.data[interior, 0:3],
        torch.tensor(
            [1.0, 0.0, 0.0], dtype=grid.dtype, device=grid.device
        ).expand(int(interior.sum().item()), 3),
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        grad_field.data[interior, 3:6],
        torch.tensor(
            [0.0, 2.0, 0.0], dtype=grid.dtype, device=grid.device
        ).expand(int(interior.sum().item()), 3),
        atol=1e-10,
        rtol=1e-10,
    )


def test_constant_field_has_zero_gradient():
    # Constant field with zero-gradient boundaries → vanishing gradient.
    grid = refined_grid(grad_scheme=GradScheme.LINEAR)
    field = CellField(grid, "const", FieldRole.LOCAL, 1)
    field.data[:] = 3.0
    zero = torch.zeros(1, dtype=grid.dtype, device=grid.device)
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
