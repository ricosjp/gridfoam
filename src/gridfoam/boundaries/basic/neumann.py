from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from jaxtyping import Float

from gridfoam.boundaries.base import BoundaryCondition
from gridfoam.boundaries.utils import get_mask
from gridfoam.meta.enums import BoundaryConditionType, FaceSide
from gridfoam.meta.types import PatchName

if TYPE_CHECKING:
    from gridfoam.core.field import CellField
else:
    CellField = Any

logger = logging.getLogger(__name__)


class NeumannBC(BoundaryCondition):
    """
    Neumann (fixed-gradient) boundary condition.

    Parameters
    ----------
    grad_value : Float[torch.Tensor, " k"]
        Fixed normal-gradient tensor prescribed at the boundary.
    """

    def __init__(self, grad_value: Float[torch.Tensor, " k"]):
        self.grad_value = grad_value

    @property
    def type(self) -> BoundaryConditionType:
        return BoundaryConditionType.NEUMANN

    def component(self, c: int) -> BoundaryCondition:
        return NeumannBC(self.grad_value[c : c + 1])

    def evaluate(
        self,
        field: CellField,
        patch_name: PatchName,
        side: FaceSide = FaceSide.UPPER,
    ) -> tuple[
        Float[torch.Tensor, " F_patch 1"],
        Float[torch.Tensor, " F_patch k"],
        Float[torch.Tensor, " F_patch k"],
    ]:
        grid = field.grid
        grad_value = self.grad_value.to(dtype=grid.dtype, device=grid.device)

        mask = get_mask(grid, patch_name, side)
        n_faces = int(mask.sum().item())

        fraction = torch.zeros(
            (n_faces, 1), dtype=grid.dtype, device=grid.device
        )
        ref_v = torch.zeros(
            (n_faces, field.num_components),
            dtype=grid.dtype,
            device=grid.device,
        )
        ref_g = (
            torch.ones(
                (n_faces, field.num_components),
                dtype=grid.dtype,
                device=grid.device,
            )
            * grad_value[None, :]
        )

        return fraction, ref_v, ref_g
