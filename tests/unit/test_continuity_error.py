"""Unit tests for continuity error metrics."""

from __future__ import annotations

from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.meta.enums import FieldRole
from gridfoam.post.diagnostics.continuity import compute_continuity_error


def test_compute_continuity_error_uniform_divergence(
    small_axis_projected_grid: AxisProjectedGrid,
) -> None:
    grid = small_axis_projected_grid
    phi = FaceField(grid, "phi", role=FieldRole.LOCAL, num_components=1)
    U = CellField(grid, "U", role=FieldRole.LOCAL, num_components=3)
    U.data.fill_(1.0)
    phi.single_data.fill_(0.01)

    local_error, global_error = compute_continuity_error(
        phi, grid.cell_volumes, grid.dt
    )

    assert local_error >= 0.0
    assert isinstance(global_error, float)


def test_compute_continuity_error_zero_flux_is_zero(
    small_axis_projected_grid: AxisProjectedGrid,
) -> None:
    grid = small_axis_projected_grid
    phi = FaceField(grid, "phi", role=FieldRole.LOCAL, num_components=1)
    phi.single_data.zero_()

    local_error, global_error = compute_continuity_error(
        phi, grid.cell_volumes, grid.dt
    )

    assert local_error == 0.0
    assert global_error == 0.0
