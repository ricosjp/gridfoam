from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from jaxtyping import Float
from torch._tensor import Tensor

from gridfoam.boundaries.base import BoundaryCondition
from gridfoam.boundaries.utils import get_mask
from gridfoam.core.shapes import require_shape
from gridfoam.meta.enums import BoundaryConditionType, FaceSide
from gridfoam.meta.types import PatchName

if TYPE_CHECKING:
    from gridfoam.core.field import CellField
else:
    CellField = Any

logger = logging.getLogger(__name__)


class DirichletBC(BoundaryCondition):
    """
    Dirichlet (fixed-value) boundary condition.

    Parameters
    ----------
    value : Float[torch.Tensor, " *component_shape"]
        Fixed value tensor prescribed at the boundary.
    """

    def __init__(self, value: Float[torch.Tensor, " *component_shape"]):
        self.value = value

    @property
    def type(self) -> BoundaryConditionType:
        return BoundaryConditionType.DIRICHLET

    @property
    def assignable(self) -> bool:
        return False

    def evaluate(
        self,
        field: CellField,
        patch_name: PatchName,
        side: FaceSide = FaceSide.UPPER,
    ) -> tuple[
        Float[torch.Tensor, " F_patch"],
        Float[torch.Tensor, " F_patch *component_shape"],
        Float[torch.Tensor, " F_patch *component_shape"],
    ]:
        grid = field.grid
        value: Tensor = self.value.to(dtype=grid.dtype, device=grid.device)

        require_shape(value, field.component_shape, "dirichlet value")
        mask = get_mask(grid, patch_name, side)
        n_faces = int(mask.sum().item())

        fraction = torch.ones((n_faces,), dtype=grid.dtype, device=grid.device)
        ref_v = (
            torch.ones(
                (n_faces, *field.component_shape),
                dtype=grid.dtype,
                device=grid.device,
            )
            * value
        )
        ref_g = torch.zeros_like(ref_v)

        return fraction, ref_v, ref_g
