"""Synthetic fields used across FV discretization tests."""

from __future__ import annotations

import torch

from gridfoam.boundaries.base import BoundaryCondition
from gridfoam.boundaries.basic.neumann import NeumannBC
from gridfoam.core.field import CellField
from gridfoam.core.grid.base import GridBase
from gridfoam.meta.enums import DomainBoundaryPatch, FieldRole
from gridfoam.meta.types import PatchName


def linear_scalar_field(
    grid: GridBase,
    *,
    name: str = "psi",
    gradient: tuple[float, float, float] = (2.0, -3.0, 5.0),
    offset: float = 7.0,
) -> tuple[CellField, torch.Tensor]:
    """
    Build a scalar field whose value is linear in cell-centre coordinates.

    ``psi(x) = gradient · x + offset``

    Domain patches receive Neumann BCs with the analytic normal derivative so
    Green-Gauss face assembly stays consistent with the linear field.
    """
    field = CellField(
        grid,
        name=name,
        role=FieldRole.LOCAL,
        component_shape=(),
    )
    grad_vec = torch.tensor(gradient, dtype=grid.dtype, device=grid.device)
    field.data = grid.cell_centers @ grad_vec + offset

    bcs: dict[PatchName, BoundaryCondition] = {}
    for patch in DomainBoundaryPatch:
        direction = patch.to_direction()
        axis = direction.value // 2
        sign = 2.0 * (direction.value % 2) - 1.0
        normal_grad = torch.tensor(
            sign * gradient[axis], dtype=grid.dtype, device=grid.device
        )
        bcs[patch] = NeumannBC(normal_grad)
    field.add_boundary_conditions(bcs)
    return field, grad_vec


def interior_mask(grid: GridBase) -> torch.Tensor:
    """Mask of non-domain-boundary cells on ``grid``."""
    mask = torch.ones(grid.num_cells, dtype=torch.bool, device=grid.device)
    mask[grid.domain_bnd_owner] = False
    return mask
