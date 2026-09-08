from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from jaxtyping import Float

from gridfoam.boundaries.base import BoundaryCondition
from gridfoam.boundaries.utils import get_mask_and_size
from gridfoam.core.shapes import require_shape
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
    grad_value : torch.Tensor
        Uniform outward normal gradient with shape ``component_shape``;
        a scalar uses shape ``()``. Boundary values extrapolate from the
        adjacent cell in both flux directions.
    """

    def __init__(self, grad_value: Float[torch.Tensor, " *component_shape"]):
        self.grad_value = grad_value

    @property
    def type(self) -> BoundaryConditionType:
        return BoundaryConditionType.NEUMANN

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
        grad_value = self.grad_value.to(dtype=grid.dtype, device=grid.device)

        require_shape(grad_value, field.component_shape, "neumann value")
        _, n_faces = get_mask_and_size(grid, patch_name, side)

        fraction = torch.zeros((n_faces,), dtype=grid.dtype, device=grid.device)
        ref_v = torch.zeros(
            (n_faces, *field.component_shape),
            dtype=grid.dtype,
            device=grid.device,
        )
        ref_g = (
            torch.ones(
                (n_faces, *field.component_shape),
                dtype=grid.dtype,
                device=grid.device,
            )
            * grad_value
        )

        return fraction, ref_v, ref_g
