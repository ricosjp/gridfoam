import logging

import torch
from jaxtyping import Bool

from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.meta.enums import DomainBoundaryPatch, FaceSide

logger = logging.getLogger(__name__)


def get_mask(
    grid: IGridBase,
    patch_name: str | DomainBoundaryPatch,
    side: FaceSide = FaceSide.UPPER,
) -> Bool[torch.Tensor, " F"]:
    if isinstance(patch_name, DomainBoundaryPatch):
        dir_id = patch_name.to_direction().value
        mask = grid.domain_bnd_dir_id == dir_id
        return mask

    if isinstance(grid, AxisProjectedGrid):
        patch_id = grid.patch_name_to_id.get(patch_name, None)
        if patch_id is None:
            logger.warning(
                "patch name: %s is not found. "
                "skipping boundary condition evaluation.",
                patch_name,
            )
            return torch.zeros(
                grid.num_immersed_faces, dtype=grid.dtype, device=grid.device
            ).bool()
        match side:
            case FaceSide.UPPER:
                return grid.ap_owner_bnd_patch_id == patch_id
            case FaceSide.LOWER:
                return grid.ap_neighbour_bnd_patch_id == patch_id
            case _:
                raise ValueError(f"Invalid side: {side}")

    raise NotImplementedError("IBM for this grid type is not supported yet.")
