"""
Flux balancing adjusts available outlets and leaves fully prescribed
outflow unchanged.
"""

from __future__ import annotations

import pathlib

import torch
from tests.helpers import channel_config

from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.boundaries.derived.inlet_outlet import InletOutletBC
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv.adjust_phi import adjust_phi
from gridfoam.meta.enums import DomainBoundaryPatch, FieldRole


def test_adjust_phi_balances_domain_flux(tmp_path: pathlib.Path) -> None:
    """
    When an adjustable outlet exists, inlet/outlet flux imbalance must be
    removed so the sum of domain-boundary phi is (near) zero.
    """
    grid = create_grid(channel_config(tmp_path))
    U = CellField(grid, "U", role=FieldRole.LOCAL, component_shape=(3,))
    phi = FaceField(grid, "phi", role=FieldRole.LOCAL, component_shape=())

    inlet_mask = grid.get_domain_bnd_mask(DomainBoundaryPatch.X_MINUS)
    outlet_mask = grid.get_domain_bnd_mask(DomainBoundaryPatch.X_PLUS)
    phi.domain_bnd_data[inlet_mask] = -1.0
    phi.domain_bnd_data[outlet_mask] = 0.8

    inlet_velocity = torch.tensor(
        [1.0, 0.0, 0.0], dtype=grid.dtype, device=grid.device
    )
    outlet_velocity = torch.tensor(
        [0.0, 0.0, 0.0], dtype=grid.dtype, device=grid.device
    )
    U.add_boundary_conditions(
        {
            DomainBoundaryPatch.X_MINUS: DirichletBC(inlet_velocity),
            DomainBoundaryPatch.X_PLUS: InletOutletBC(outlet_velocity),
        }
    )

    adjust_phi(phi, U)

    total = phi.domain_bnd_data.sum().item()
    assert abs(total) < 1e-10


def test_adjust_phi_skips_scaling_when_no_adjustable_outflow(
    tmp_path: pathlib.Path,
) -> None:
    """
    With only fixed Dirichlet outlets, adjust_phi must leave phi unchanged.
    """
    grid = create_grid(channel_config(tmp_path))
    U = CellField(grid, "U", role=FieldRole.LOCAL, component_shape=(3,))
    phi = FaceField(grid, "phi", role=FieldRole.LOCAL, component_shape=())

    inlet_mask = grid.get_domain_bnd_mask(DomainBoundaryPatch.X_MINUS)
    outlet_mask = grid.get_domain_bnd_mask(DomainBoundaryPatch.X_PLUS)
    phi.domain_bnd_data[inlet_mask] = -1.0
    phi.domain_bnd_data[outlet_mask] = 0.5

    inlet_velocity = torch.tensor(
        [1.0, 0.0, 0.0], dtype=grid.dtype, device=grid.device
    )
    outlet_velocity = torch.tensor(
        [0.0, 0.0, 0.0], dtype=grid.dtype, device=grid.device
    )
    U.add_boundary_conditions(
        {
            DomainBoundaryPatch.X_MINUS: DirichletBC(inlet_velocity),
            DomainBoundaryPatch.X_PLUS: DirichletBC(outlet_velocity),
        }
    )

    phi_before = phi.domain_bnd_data.clone()
    adjust_phi(phi, U)
    assert torch.allclose(phi.domain_bnd_data, phi_before)
