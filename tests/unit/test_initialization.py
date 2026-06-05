from __future__ import annotations

import torch
from tests.conftest import small_gridfoam_config

from gridfoam.core.field import CellField
from gridfoam.core.grid.factory import create_grid
from gridfoam.initialization import initialize_from_dirichlet_patch
from gridfoam.meta.config import BoundaryConditionConfig
from gridfoam.meta.enums import (
    BoundaryConditionType,
    DomainBoundaryPatch,
    FieldRole,
)


def test_initialize_from_dirichlet_patch_sets_cell_field():
    grid = create_grid(small_gridfoam_config())
    field = CellField(grid, "U_init", FieldRole.LOCAL, num_components=3)
    bcs = [
        BoundaryConditionConfig(
            name="inlet",
            type=BoundaryConditionType.DIRICHLET,
            patches=[DomainBoundaryPatch.X_MINUS],
            value=[1.0, 0.0, 0.0],
        )
    ]

    initialized = initialize_from_dirichlet_patch(
        field, bcs, DomainBoundaryPatch.X_MINUS
    )

    assert initialized
    expected = torch.tensor([1.0, 0.0, 0.0], dtype=grid.dtype)
    torch.testing.assert_close(field.data, expected.expand_as(field.data))


def test_initialize_from_dirichlet_patch_leaves_field_without_patch():
    grid = create_grid(small_gridfoam_config())
    field = CellField(grid, "U_init_none", FieldRole.LOCAL, num_components=3)
    bcs = [
        BoundaryConditionConfig(
            name="wall",
            type=BoundaryConditionType.DIRICHLET,
            patches=[DomainBoundaryPatch.Y_MINUS],
            value=[0.0, 0.0, 0.0],
        )
    ]

    initialized = initialize_from_dirichlet_patch(
        field, bcs, DomainBoundaryPatch.X_MINUS
    )

    assert not initialized
    torch.testing.assert_close(field.data, torch.zeros_like(field.data))
