"""Unit tests for ``IGridBase.to``."""

from __future__ import annotations

import pytest
import torch
from tests.conftest import small_gridfoam_config

from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.factory import create_grid
from gridfoam.meta.enums import DeviceType, FieldRole


def _fresh_grid() -> AxisProjectedGrid:
    grid = create_grid(small_gridfoam_config())
    assert isinstance(grid, AxisProjectedGrid)
    return grid


def test_to_same_device_returns_same_object() -> None:
    # Moving to the current device is a no-op on identity.
    grid = _fresh_grid()
    assert grid.to("cpu") is grid
    assert grid.to(torch.device("cpu")) is grid


def test_to_preserves_topology_scalars() -> None:
    # Device moves must not change topology counts.
    grid = _fresh_grid()
    n_cells = grid.num_cells
    n_faces = grid.num_internal_faces
    n_bnd = grid.num_domain_bnd_faces
    n_ib = grid.num_immersed_faces
    grid.to("cpu")
    assert grid.num_cells == n_cells
    assert grid.num_internal_faces == n_faces
    assert grid.num_domain_bnd_faces == n_bnd
    assert grid.num_immersed_faces == n_ib


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_to_cuda_moves_geometry_and_registered_fields() -> None:
    # Geometry and registered field buffers share the runtime device.
    grid = _fresh_grid()
    u = CellField(grid, "U", FieldRole.LOCAL, 3)
    phi = FaceField(grid, "phi", FieldRole.LOCAL, 1)
    u.data[:] = 1.25
    phi.single_data[:] = 0.5
    u_cpu = u.data.detach().cpu().clone()
    phi_cpu = phi.single_data.detach().cpu().clone()
    owner_cpu = grid.owner.detach().cpu().clone()
    sim_device = grid.sim_config.device

    assert grid.to("cuda") is grid
    assert grid.device.type == "cuda"
    assert u.data.device == grid.device
    assert u.old_data is u.data
    assert phi.single_data.device == grid.device
    assert phi.domain_bnd_data.device == grid.device
    assert phi.single_mask.device == grid.device
    assert grid.owner.device == grid.device
    assert torch.equal(u.data.cpu(), u_cpu)
    assert torch.equal(phi.single_data.cpu(), phi_cpu)
    assert torch.equal(grid.owner.cpu(), owner_cpu)
    assert grid.sim_config.device == sim_device
    assert sim_device == DeviceType.CPU
    assert grid.to("cuda") is grid
