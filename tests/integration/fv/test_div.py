"""
Integration tests for the cell-centered divergence operator (``fvc.div``).

Pins the OpenFOAM ``fvc::div`` convention: the face sum is normalized by
cell volume, so a linear velocity field yields its analytic divergence on
refined octree meshes.
"""

from __future__ import annotations

import torch
from tests.helpers import refined_grid

from gridfoam.core.field import FaceField
from gridfoam.core.grid.base import IGridBase
from gridfoam.fv import fvc
from gridfoam.meta.enums import FieldRole


def _uniform_divergence_flux(grid: IGridBase) -> FaceField:
    """Face flux of ``U = (x, 0, 0)``, whose analytic divergence is one."""

    def flux(face_centers: torch.Tensor, Sf: torch.Tensor) -> torch.Tensor:
        U = torch.zeros_like(face_centers)
        U[:, 0] = face_centers[:, 0]
        return torch.sum(U * Sf, dim=1, keepdim=True)

    phi = FaceField(grid, "phi", FieldRole.LOCAL, 1)
    single_mask = phi.single_mask
    phi.single_data = flux(grid.face_centers[single_mask], grid.Sf[single_mask])
    phi.domain_bnd_data = flux(grid.domain_bnd_face_centers, grid.domain_bnd_Sf)
    return phi


def test_div_is_normalized_by_cell_volume():
    # Without volume normalization the face sum would scale with cell volume,
    # which varies across octree levels on this grid.
    grid = refined_grid()
    phi = _uniform_divergence_flux(grid)

    div_phi = fvc.div(phi)

    expected = torch.ones_like(div_phi.data)
    torch.testing.assert_close(div_phi.data, expected, atol=1e-12, rtol=1e-12)
    assert grid.cell_volumes.max() > 2.0 * grid.cell_volumes.min()


def test_div_of_divergence_free_flux_vanishes():
    # Uniform velocity ``U = (1, 0, 0)`` → ``phi = Sf_x`` has zero divergence.
    grid = refined_grid()
    phi = FaceField(grid, "phi_uniform", FieldRole.LOCAL, 1)
    single_mask = phi.single_mask
    phi.single_data = grid.Sf[single_mask][:, :1]
    phi.domain_bnd_data = grid.domain_bnd_Sf[:, :1]

    div_phi = fvc.div(phi)

    torch.testing.assert_close(
        div_phi.data, torch.zeros_like(div_phi.data), atol=1e-12, rtol=1e-12
    )
