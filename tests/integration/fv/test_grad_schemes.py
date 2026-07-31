"""
Integration tests for cell-centre gradient schemes on refined meshes.

Verifies public ``fvc.grad`` contracts, hierarchy accuracy, and IB-aware
least-squares stencils.
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
from gridfoam.fv.kernels.face_interpolation import (
    linear_internal_face_values,
    single_internal_mask,
)
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


def test_linear_grad_is_more_accurate_than_uncorrected_gauss_on_refined_mesh():
    # LINEAR grad must beat Green-Gauss assembled from uncorrected linear
    # face values on a 3-D refined mesh.
    grid = refined_3d_grid(GradScheme.LINEAR)
    field, expected_vec = linear_scalar_field(grid)
    interior = interior_mask(grid)
    expected = expected_vec.to(grid.device)

    uncorrected_faces = fvc.interpolate(field)
    uncorrected_faces.single_data = linear_internal_face_values(field)
    uncorrected = assemble_gauss_gradient(grid, uncorrected_faces)[:, 0, :]
    corrected = fvc.grad(field).data

    uncorrected_err = torch.linalg.vector_norm(
        uncorrected[interior] - expected
    ).item()
    corrected_err = torch.linalg.vector_norm(
        corrected[interior] - expected
    ).item()
    assert corrected_err < uncorrected_err


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
    # A constant field with zero-gradient boundaries must produce a vanishing
    # gradient everywhere.
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


def test_leastsquare_uses_only_single_sided_internal_faces():
    # Immersed faces must not enter the least-squares owner/neighbour stencil.
    # On a mesh without immersed geometry this reduces to verifying that the
    # single-sided mask is the full internal set used by the scheme path.
    grid = refined_grid(grad_scheme=GradScheme.LEASTSQUARE)
    single_mask = single_internal_mask(grid)
    assert int(single_mask.sum().item()) == grid.num_internal_faces - getattr(
        grid, "num_immersed_faces", 0
    )
    field, expected = linear_scalar_field(grid)
    grad_field = fvc.grad(field)
    interior = interior_mask(grid)
    torch.testing.assert_close(
        grad_field.data[interior],
        expected.expand_as(grad_field.data[interior]),
        atol=1e-10,
        rtol=1e-10,
    )
