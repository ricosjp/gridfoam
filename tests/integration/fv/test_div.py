"""
Integration tests for the cell-centered divergence operator (``fvc.div``).

Pins the OpenFOAM ``fvc::div`` convention: the face sum is normalized by the
cell volume, so a linear velocity field yields its analytic divergence
independently of the local refinement level.
"""

from __future__ import annotations

import torch
from tests.helpers import refined_grid

from gridfoam.core.field import FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
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
    if isinstance(grid, AxisProjectedGrid) and grid.num_immersed_faces > 0:
        immersed = grid.ap_is_immersed_faces
        immersed_flux = flux(grid.face_centers[immersed], grid.Sf[immersed])
        phi.immersed_upper = immersed_flux.clone()
        phi.immersed_lower = immersed_flux.clone()
    return phi


def test_div_is_normalized_by_cell_volume():
    # ``fvc.div`` must reproduce the analytic divergence of a linear field.
    # A face sum without volume normalization would instead scale with the
    # cell volume, which varies across octree levels on this grid.
    grid = refined_grid()
    phi = _uniform_divergence_flux(grid)

    div_phi = fvc.div(phi)

    expected = torch.ones_like(div_phi.data)
    torch.testing.assert_close(div_phi.data, expected, atol=1e-12, rtol=1e-12)
    assert grid.cell_volumes.max() > 2.0 * grid.cell_volumes.min()


def test_div_of_divergence_free_flux_vanishes():
    # A uniform velocity field carries no divergence on any refinement level.
    grid = refined_grid()
    phi = FaceField(grid, "phi_uniform", FieldRole.LOCAL, 1)
    single_mask = phi.single_mask
    phi.single_data = grid.Sf[single_mask][:, :1]
    phi.domain_bnd_data = grid.domain_bnd_Sf[:, :1]

    div_phi = fvc.div(phi)

    expected = torch.zeros_like(div_phi.data)
    torch.testing.assert_close(div_phi.data, expected, atol=1e-12, rtol=1e-12)
