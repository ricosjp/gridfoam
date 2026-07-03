"""Synthetic fields used across FV discretization tests."""

from __future__ import annotations

import torch

from gridfoam.core.field import CellField
from gridfoam.core.grid.base import IGridBase
from gridfoam.meta.enums import FieldRole


def linear_scalar_field(
    grid: IGridBase,
    *,
    name: str = "psi",
    gradient: tuple[float, float, float] = (2.0, -3.0, 5.0),
    offset: float = 7.0,
) -> tuple[CellField, torch.Tensor]:
    """
    Build a scalar field whose value is linear in cell-centre coordinates.

    ``psi(x) = gradient · x + offset``
    """
    field = CellField(
        grid,
        name=name,
        role=FieldRole.LOCAL,
        num_components=1,
    )
    grad_vec = torch.tensor(gradient, dtype=grid.dtype, device=grid.device)
    field.data = (grid.cell_centers @ grad_vec + offset).reshape(-1, 1)
    return field, grad_vec


def interior_mask(grid: IGridBase) -> torch.Tensor:
    """Mask of non-domain-boundary cells on ``grid``."""
    mask = torch.ones(grid.num_cells, dtype=torch.bool, device=grid.device)
    mask[grid.domain_bnd_owner] = False
    return mask
