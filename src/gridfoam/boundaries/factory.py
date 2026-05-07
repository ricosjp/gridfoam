from __future__ import annotations

import torch

from gridfoam.boundaries.base import BoundaryCondition
from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.boundaries.basic.empty import EmptyBC
from gridfoam.boundaries.basic.neumann import NeumannBC
from gridfoam.boundaries.basic.slip import SlipBC
from gridfoam.boundaries.derived.fixed_flux_pressure import FixedFluxPressure
from gridfoam.boundaries.derived.inlet_outlet import InletOutletBC
from gridfoam.core.field import CellField
from gridfoam.meta.config import BoundaryConditionConfig
from gridfoam.meta.enums import BoundaryConditionType
from gridfoam.meta.types import PatchName


def _to_tensor(
    values: list[float] | None, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if values is None:
        raise ValueError("Boundary condition value is required.")
    tensor = torch.tensor(values, dtype=dtype, device=device)
    return tensor.reshape(-1)


def create_boundary_condition(
    bc_config: BoundaryConditionConfig, dtype: torch.dtype, device: torch.device
) -> BoundaryCondition:
    bc_type = bc_config.type

    if bc_type == BoundaryConditionType.DIRICHLET:
        return DirichletBC(_to_tensor(bc_config.value, dtype, device))
    if bc_type == BoundaryConditionType.NEUMANN:
        return NeumannBC(_to_tensor(bc_config.value, dtype, device))
    if bc_type == BoundaryConditionType.EMPTY:
        return EmptyBC()
    if bc_type == BoundaryConditionType.SLIP:
        return SlipBC()
    if bc_type == BoundaryConditionType.INLET_OUTLET:
        return InletOutletBC(
            inlet_value=_to_tensor(bc_config.value, dtype, device),
            phi_builtin_key=bc_config.phi_builtin_key,
        )
    if bc_type == BoundaryConditionType.FIXED_FLUX_PRESSURE:
        return FixedFluxPressure(
            phi_builtin_key=bc_config.phi_builtin_key,
            HbyA_builtin_key=bc_config.HbyA_builtin_key,
            rAU_builtin_key=bc_config.rAU_builtin_key,
        )
    raise ValueError(f"Unsupported boundary condition type: {bc_type}")


def apply_boundary_condition_configs(
    field: CellField, bc_configs: list[BoundaryConditionConfig]
) -> None:
    bcs: dict[PatchName, BoundaryCondition] = {}
    for bc_config in bc_configs:
        bc = create_boundary_condition(
            bc_config,
            dtype=field.grid.dtype,
            device=field.grid.device,
        )
        for patch in bc_config.patches:
            bcs[patch] = bc
    field.add_boundary_conditions(bcs)
