"""Unit tests for ``NewtonianTransport.nu``."""

from __future__ import annotations

import pytest
import torch
from tests.conftest import small_gridfoam_config

from gridfoam.core.grid.factory import create_grid
from gridfoam.models.transport.newtonian import NewtonianTransport
from gridfoam.models.turbulence.laminar import Laminar


def _transport() -> tuple[NewtonianTransport, float]:
    grid = create_grid(small_gridfoam_config())
    nu_cfg = float(grid.sim_config.properties.transport.nu)
    return NewtonianTransport(grid), nu_cfg


def test_nu_returns_tensor_from_config() -> None:
    # Case-file float is stored and returned as a tensor of shape [1].
    transport, nu_cfg = _transport()
    nu = transport.nu()
    assert isinstance(nu, torch.Tensor)
    assert nu.shape == ()
    assert nu.item() == pytest.approx(nu_cfg)
    assert nu.dtype == transport.grid.dtype
    assert nu.device == transport.grid.device


def test_nu_same_device_preserves_identity() -> None:
    # Matching device and dtype must not copy the stored tensor.
    transport, _ = _transport()
    nu = transport.nu()
    assert transport.nu() is nu


def test_set_nu_tensor_is_returned_as_same_object() -> None:
    # Autograd leaves stay the same object when already on the grid.
    transport, _ = _transport()
    nu = torch.tensor(
        0.05,
        dtype=transport.grid.dtype,
        device=transport.grid.device,
        requires_grad=True,
    )
    transport.set_nu(nu)
    assert transport.nu() is nu


def test_set_nu_float_wraps_as_tensor() -> None:
    transport, _ = _transport()
    transport.set_nu(0.02)
    nu = transport.nu()
    assert isinstance(nu, torch.Tensor)
    assert nu.shape == ()
    assert nu.item() == pytest.approx(0.02)


def test_nu_flows_into_nu_eff_autograd() -> None:
    # nu_eff = nu + nu_t must keep the viscosity leaf in the graph.
    grid = create_grid(small_gridfoam_config())
    laminar = Laminar(grid)
    nu = torch.tensor(
        0.1,
        dtype=grid.dtype,
        device=grid.device,
        requires_grad=True,
    )
    assert isinstance(laminar.transport, NewtonianTransport)
    laminar.transport.set_nu(nu)
    loss = laminar.nu_eff().sum()
    loss.backward()
    assert nu.grad is not None
    assert torch.allclose(nu.grad, torch.full_like(nu, grid.num_cells))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nu_follows_grid_device_after_to() -> None:
    # Runtime device comes from the grid; sim_config.device is unchanged.
    transport, _ = _transport()
    grid = transport.grid
    grid.to("cuda")
    nu = transport.nu()
    assert nu.device == grid.device
    assert nu.device.type == "cuda"
