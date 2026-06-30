from __future__ import annotations

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


class EmptyBC(BoundaryCondition):
    """
    Empty (2D) boundary condition.

    This boundary type marks faces that should be excluded from
    3D operators when solving an effectively 2D case.
    """

    @property
    def type(self) -> BoundaryConditionType:
        return BoundaryConditionType.EMPTY

    def component(self, c: int) -> BoundaryCondition:
        return self

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
        ref_g = torch.zeros_like(ref_v)
        return fraction, ref_v, ref_g
