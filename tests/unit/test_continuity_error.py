"""
Continuity diagnostics use dt-scaled volume averages of signed and absolute
divergence.
"""

from __future__ import annotations

import pytest
import torch

from gridfoam.core.field import FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv import fvc
from gridfoam.meta.enums import FieldRole
from gridfoam.post.diagnostics.continuity import compute_continuity_error


def test_compute_continuity_error_matches_volume_weighted_div(
    small_axis_projected_grid: AxisProjectedGrid,
) -> None:
    """
    Local and global errors use absolute and signed volume-weighted
    divergence times dt.
    """
    grid = small_axis_projected_grid
    phi = FaceField(grid, "phi", role=FieldRole.LOCAL, component_shape=())
    phi.single_data.fill_(0.01)
    phi.domain_bnd_data.fill_(0.01)

    volumes = grid.cell_volumes
    cont_err = fvc.div(phi).data
    total_volume = torch.sum(volumes)
    expected_local = (
        grid.dt * torch.sum(torch.abs(cont_err) * volumes) / total_volume
    ).item()
    expected_global = (
        grid.dt * torch.sum(cont_err * volumes) / total_volume
    ).item()

    local_error, global_error = compute_continuity_error(phi, volumes, grid.dt)

    assert local_error == pytest.approx(expected_local)
    assert global_error == pytest.approx(expected_global)


def test_compute_continuity_error_zero_flux_is_zero(
    small_axis_projected_grid: AxisProjectedGrid,
) -> None:
    """Zero face flux gives exactly zero local and global continuity error."""
    grid = small_axis_projected_grid
    phi = FaceField(grid, "phi", role=FieldRole.LOCAL, component_shape=())
    phi.single_data.zero_()
    phi.domain_bnd_data.zero_()

    local_error, global_error = compute_continuity_error(
        phi, grid.cell_volumes, grid.dt
    )

    assert local_error == 0.0
    assert global_error == 0.0
