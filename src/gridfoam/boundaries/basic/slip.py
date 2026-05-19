from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.boundaries.base import BoundaryCondition
from gridfoam.boundaries.utils import get_mask
from gridfoam.core.field import CellField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.meta.enums import (
    BoundaryConditionType,
    DomainBoundaryPatch,
    FaceSide,
)
from gridfoam.meta.types import PatchName


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

        fraction = torch.ones(
            (n_faces, 1), dtype=grid.dtype, device=grid.device
        )
        ref_v = torch.zeros(
            (n_faces, field.num_components),
            dtype=grid.dtype,
            device=grid.device,
        )
        ref_g = torch.zeros(
            (n_faces, field.num_components),
            dtype=grid.dtype,
            device=grid.device,
        )
        # For scalar fields (e.g., pressure), use simple zero-gradient Neumann.
        if field.num_components == 1:
            return fraction, ref_v, ref_g

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

        psi_O = field.data[target_cells]  # [F_patch k]
        psi_O_dot_n = torch.sum(
            psi_O * n_vec, dim=1, keepdim=True
        )  # [F_patch 1]

        ref_v = psi_O - psi_O_dot_n * n_vec

        return fraction, ref_v, ref_g
