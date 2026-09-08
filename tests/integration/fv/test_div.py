"""
Cell divergence matches analytic flux balance across octree refinement.

Volume normalization gives unit divergence for U=(x, 0, 0), despite unequal
cell sizes. The face flux of a uniform velocity has zero divergence.
"""

from __future__ import annotations

import torch
from tests.helpers import refined_grid

from gridfoam.core.field import FaceField
from gridfoam.core.grid.base import GridBase
from gridfoam.fv import fvc
from gridfoam.meta.enums import FieldRole


def _uniform_divergence_flux(grid: GridBase) -> FaceField:
    """Face flux of ``U = (x, 0, 0)``, whose analytic divergence is one."""

    def flux(face_centers: torch.Tensor, Sf: torch.Tensor) -> torch.Tensor:
        U = torch.zeros_like(face_centers)
        U[:, 0] = face_centers[:, 0]
        return torch.sum(U * Sf, dim=1, keepdim=False)

    phi = FaceField(grid, "phi", FieldRole.LOCAL, ())
    single_mask = phi.single_mask
    phi.single_data = flux(grid.face_centers[single_mask], grid.Sf[single_mask])
    phi.domain_bnd_data = flux(grid.domain_bnd_face_centers, grid.domain_bnd_Sf)
    return phi


def test_div_is_normalized_by_cell_volume() -> None:
    """
    The flux of U=(x, 0, 0) gives unit divergence across unequal cell
    volumes.
    """
    grid = refined_grid()
    phi = _uniform_divergence_flux(grid)

    div_phi = fvc.div(phi)

    expected = torch.ones_like(div_phi.data)
    torch.testing.assert_close(div_phi.data, expected, atol=1e-12, rtol=1e-12)
    assert grid.cell_volumes.max() > 2.0 * grid.cell_volumes.min()


def test_div_of_divergence_free_flux_vanishes() -> None:
    """
    Uniform velocity ``U = (1, 0, 0)`` → ``phi = Sf_x`` has zero
    divergence.
    """
    grid = refined_grid()
    phi = FaceField(grid, "phi_uniform", FieldRole.LOCAL, ())
    single_mask = phi.single_mask
    phi.single_data = grid.Sf[single_mask][:, 0]
    phi.domain_bnd_data = grid.domain_bnd_Sf[:, 0]

    div_phi = fvc.div(phi)

    torch.testing.assert_close(
        div_phi.data, torch.zeros_like(div_phi.data), atol=1e-12, rtol=1e-12
    )
