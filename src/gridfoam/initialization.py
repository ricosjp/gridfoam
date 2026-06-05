from collections.abc import Iterable

import torch

from gridfoam.core.field import CellField
from gridfoam.meta.config import BoundaryConditionConfig
from gridfoam.meta.enums import BoundaryConditionType, DomainBoundaryPatch


def initialize_from_dirichlet_patch(
    field: CellField,
    bc_configs: Iterable[BoundaryConditionConfig],
    patch: DomainBoundaryPatch,
) -> bool:
    """
    Initialize a cell field from a fixed-value domain boundary.

    This is useful for steady external-flow cases, where starting the whole
    domain from zero velocity can leave large refined regions under-filled for
    many SIMPLE pseudo-iterations.
    """
    for bc_config in bc_configs:
        if bc_config.type != BoundaryConditionType.DIRICHLET:
            continue
        if patch not in bc_config.patches:
            continue
        if bc_config.value is None:
            raise ValueError(
                f"Dirichlet boundary {bc_config.name!r} has no value."
            )
        value = torch.tensor(
            bc_config.value, dtype=field.grid.dtype, device=field.grid.device
        )
        if value.numel() != field.num_components:
            raise ValueError(
                f"Cannot initialize {field.name!r}: boundary value has "
                f"{value.numel()} components, expected {field.num_components}."
            )
        field.data[:] = value.reshape(1, -1)
        return True
    return False
