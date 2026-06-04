from __future__ import annotations

import logging

import torch
from jaxtyping import Float

from gridfoam.boundaries.base import BoundaryCondition
from gridfoam.boundaries.utils import get_mask
from gridfoam.core.field import CellField
from gridfoam.meta.enums import BoundaryConditionType, FaceSide
from gridfoam.meta.types import PatchName

logger = logging.getLogger(__name__)


class DirichletBC(BoundaryCondition):
    """
    Dirichlet (fixed-value) boundary condition.

    Parameters
    ----------
    value : Float[torch.Tensor, " k"]
        Fixed value tensor prescribed at the boundary.
    """

    def __init__(self, value: Float[torch.Tensor, " k"]):
        self.value = value

    @property
    def type(self) -> BoundaryConditionType:
        return BoundaryConditionType.DIRICHLET

    def component(self, c: int) -> BoundaryCondition:
        return DirichletBC(self.value[c : c + 1])

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
        value = self.value.to(dtype=grid.dtype, device=grid.device)

        mask = get_mask(grid, patch_name, side)
        n_faces = int(mask.sum().item())

        fraction = torch.ones(
            (n_faces, 1), dtype=grid.dtype, device=grid.device
        )
        ref_v = (
            torch.ones(
                (n_faces, field.num_components),
                dtype=grid.dtype,
                device=grid.device,
            )
            * value[None, :]
        )
        ref_g = torch.zeros_like(ref_v)

        return fraction, ref_v, ref_g
