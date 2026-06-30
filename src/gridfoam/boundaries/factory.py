from __future__ import annotations

import torch

from gridfoam.boundaries.base import BoundaryCondition
from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.boundaries.basic.empty import EmptyBC
from gridfoam.boundaries.basic.neumann import NeumannBC
from gridfoam.boundaries.basic.slip import SlipBC
from gridfoam.boundaries.derived.fixed_flux_pressure import FixedFluxPressure
from gridfoam.boundaries.derived.inlet_outlet import InletOutletBC
from gridfoam.meta.config import BoundaryConditionConfig
from gridfoam.meta.enums import BoundaryConditionType


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
            phase=bc_config.phase,
        )
    if bc_type == BoundaryConditionType.FIXED_FLUX_PRESSURE:
        return FixedFluxPressure(
            phase=bc_config.phase,
        )
    raise ValueError(f"Unsupported boundary condition type: {bc_type}")
