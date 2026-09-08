"""
Face flux reconstructs uniform and linear velocities across octree refinement.

Uniform velocity is recovered in every cell and written into the target field.
The linear U=(x, 2y, -3z) case checks component values on a refined 3-D grid.
"""

from __future__ import annotations

import torch
from tests.helpers import refined_3d_grid, refined_grid

from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.base import GridBase
from gridfoam.fv import fvc
from gridfoam.meta.enums import FieldRole, GradScheme


def _uniform_case(
    grid: GridBase, U_ref: tuple[float, float, float]
) -> tuple[FaceField, torch.Tensor]:
    U_vec = torch.tensor(U_ref, dtype=grid.dtype, device=grid.device)
    phi = FaceField(grid, "phi_uniform_rec", FieldRole.LOCAL, ())
    single_mask = phi.single_mask
    phi.single_data = grid.Sf[single_mask] @ U_vec
    phi.domain_bnd_data = grid.domain_bnd_Sf @ U_vec
    return phi, U_vec


def test_reconstruct_recovers_uniform_velocity_on_refined_grid() -> None:
    """
    A uniform velocity must be recovered exactly in every cell, including
    coarse cells whose faces are split into hanging-node sub-faces.
    """
    grid = refined_grid()
    phi, U_vec = _uniform_case(grid, (1.0, 0.5, -0.25))
    U = CellField(grid, "U_rec", FieldRole.LOCAL, (3,))

    u_data = fvc.reconstruct(phi, U)

    expected = U_vec.expand_as(u_data)
    torch.testing.assert_close(u_data, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(U.data, expected, atol=1e-12, rtol=1e-12)


def test_reconstruct_recovers_linear_velocity_on_3d_refined_grid() -> None:
    """
    ``U = (x, 2y, -3z)`` is linear, so the area-weighted average of the
    face-normal velocities equals the cell-centre value on Cartesian cells.
    """
    grid = refined_3d_grid(GradScheme.LEASTSQUARE)
    scale = torch.tensor([1.0, 2.0, -3.0], dtype=grid.dtype, device=grid.device)

    phi = FaceField(grid, "phi_linear_rec", FieldRole.LOCAL, ())
    single_mask = phi.single_mask
    U_faces = grid.face_centers[single_mask] * scale
    phi.single_data = torch.sum(
        U_faces * grid.Sf[single_mask], dim=1, keepdim=False
    )
    U_bnd = grid.domain_bnd_face_centers * scale
    phi.domain_bnd_data = torch.sum(
        U_bnd * grid.domain_bnd_Sf, dim=1, keepdim=False
    )
    U = CellField(grid, "U_rec_linear", FieldRole.LOCAL, (3,))

    u_data = fvc.reconstruct(phi, U)

    expected = grid.cell_centers * scale
    torch.testing.assert_close(u_data, expected, atol=1e-12, rtol=1e-12)
