import logging

import torch
from jaxtyping import Bool

from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import GridBase
from gridfoam.meta.enums import DomainBoundaryPatch, FaceSide

logger = logging.getLogger(__name__)


def get_mask(
    grid: GridBase,
    patch_name: str | DomainBoundaryPatch,
    side: FaceSide = FaceSide.UPPER,
) -> Bool[torch.Tensor, " F"]:
    """Read-only patch mask, cached until grid geometry is invalidated."""
    return get_mask_and_size(grid, patch_name, side)[0]


def get_mask_and_size(
    grid: GridBase,
    patch_name: str | DomainBoundaryPatch,
    side: FaceSide = FaceSide.UPPER,
) -> tuple[Bool[torch.Tensor, " F"], int]:
    """Return a cached patch mask and its number of selected faces.

    The mask spans domain-boundary faces for a ``DomainBoundaryPatch`` or
    immersed faces for a named surface patch; ``side`` selects the immersed
    side. Treat the returned mask as read-only. The Python integer count is
    cached with the mask to avoid repeated GPU synchronization. Both are
    discarded when the grid invalidates its derived caches.
    """
    key = (patch_name, side)
    cache = grid.fv_cache.boundary_masks
    if key not in cache:
        mask = _build_mask(grid, patch_name, side)
        cache[key] = (mask, int(mask.sum().item()))
    return cache[key]


def _build_mask(
    grid: GridBase,
    patch_name: str | DomainBoundaryPatch,
    side: FaceSide,
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
