from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from jaxtyping import Float

from gridfoam.boundaries.base import BoundaryCondition
from gridfoam.boundaries.utils import get_mask_and_size
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.meta.enums import (
    BoundaryConditionType,
    DomainBoundaryPatch,
    FaceSide,
)
from gridfoam.meta.types import PatchName

if TYPE_CHECKING:
    from gridfoam.core.field import CellField
else:
    CellField = Any


class SlipBC(BoundaryCondition):
    """
    Slip-wall boundary condition.

    Physically, the normal velocity is zero (impermeable wall) and
    tangential velocity gradient is zero. This is represented by a
    virtual Neumann gradient computed from the target slip state.
    """

    def __init__(self):
        pass

    @property
    def type(self) -> BoundaryConditionType:
        return BoundaryConditionType.SLIP

    @property
    def assignable(self) -> bool:
        # OpenFOAM ``slipFvPatchField::assignable`` returns false: the wall
        # flux is fixed to zero, so ``constrainHbyA`` must not extrapolate.
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
        if field.component_shape not in ((), (3,)):
            raise ValueError("SlipBC supports scalar and vector fields")
        grid = field.grid

        mask, n_faces = get_mask_and_size(grid, patch_name, side)

        fraction = torch.ones((n_faces,), dtype=grid.dtype, device=grid.device)
        ref_v = torch.zeros(
            (n_faces, *field.component_shape),
            dtype=grid.dtype,
            device=grid.device,
        )
        ref_g = torch.zeros(
            (n_faces, *field.component_shape),
            dtype=grid.dtype,
            device=grid.device,
        )
        # For scalar fields (e.g., pressure), use simple zero-gradient Neumann.
        if field.component_shape == ():
            return torch.zeros_like(fraction), ref_v, ref_g

        # For vector fields (e.g., velocity), compute virtual Neumann gradient.
        if isinstance(patch_name, DomainBoundaryPatch):
            target_cells = grid.domain_bnd_owner[mask]
            Sf_bnd = grid.domain_bnd_Sf[mask]
        elif isinstance(grid, AxisProjectedGrid):
            immersed_mask = grid.ap_is_immersed_faces
            if side == FaceSide.UPPER:
                target_cells = grid.owner[immersed_mask][mask]
                Sf_bnd = grid.Sf[immersed_mask][mask]
            elif side == FaceSide.LOWER:
                target_cells = grid.neighbour[immersed_mask][mask]
                Sf_bnd = -grid.Sf[immersed_mask][mask]
            else:
                raise ValueError(f"Invalid side: {side}")
        else:
            raise NotImplementedError(
                "IBM for this grid type is not supported yet."
            )

        n_vec = torch.sign(Sf_bnd)
        psi_O = field.data[target_cells]
        normal_velocity = (psi_O * n_vec).sum(dim=-1)
        ref_v = psi_O - normal_velocity[:, None] * n_vec

        return fraction, ref_v, ref_g
