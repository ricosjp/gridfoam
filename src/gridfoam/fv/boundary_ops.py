from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum, auto
from weakref import WeakKeyDictionary

import torch
from jaxtyping import Bool, Float, Int

from gridfoam.core.field import CellField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.meta.enums import (
    BoundaryConditionType,
    DomainBoundaryPatch,
    FaceSide,
)
from gridfoam.meta.types import PatchName


class BoundaryFaceKind(StrEnum):
    DOMAIN = auto()
    IMMERSED_UPPER = auto()
    IMMERSED_LOWER = auto()


@dataclass(frozen=True)
class BoundaryBatch:
    patch_name: PatchName
    side: FaceSide
    face_kind: BoundaryFaceKind
    face_mask: Bool[torch.Tensor, " F_any"]
    target_cells: Int[torch.Tensor, " F_any"]
    mag_d: Float[torch.Tensor, " F_any 1"]


BoundaryBatchCacheKey = tuple[
    tuple[int, int, int, int, int, tuple[float, ...], tuple[float, ...]]
    | tuple[int, int],
    tuple[tuple[PatchName, int, BoundaryConditionType], ...],
]

_boundary_batch_cache: WeakKeyDictionary[
    CellField,
    tuple[BoundaryBatchCacheKey, tuple[BoundaryBatch, ...]],
] = WeakKeyDictionary()


def _grid_topology_token(
    grid: IGridBase,
) -> (
    tuple[int, int, int, int, int, tuple[float, ...], tuple[float, ...]]
    | tuple[int, int]
):
    """Fingerprint used to invalidate cached boundary batches."""
    if isinstance(grid, AxisProjectedGrid):
        return (
            grid.num_cells,
            grid.num_internal_faces,
            grid.num_immersed_faces,
            id(grid.ap_is_immersed_faces),
            id(grid.ap_dist_owner_to_bnd),
            tuple(grid.ib_translation),
            tuple(grid.ib_rotation_quaternion),
        )
    return (grid.num_cells, grid.num_internal_faces)


def _boundary_batch_cache_key(field: CellField) -> BoundaryBatchCacheKey:
    return (
        _grid_topology_token(field.grid),
        tuple((patch, id(bc), bc.type) for patch, bc in field.bcs.items()),
    )


def iter_boundary_batches(field: CellField) -> Iterator[BoundaryBatch]:
    cache_key = _boundary_batch_cache_key(field)
    cached = _boundary_batch_cache.get(field)
    if cached is not None:
        cached_key, cached_batches = cached
        if cached_key == cache_key:
            yield from cached_batches
            return

    batches = tuple(_build_boundary_batches(field))
    _boundary_batch_cache[field] = (cache_key, batches)
    yield from batches


def _build_boundary_batches(field: CellField) -> Iterator[BoundaryBatch]:
    grid = field.grid
    for patch, bc in field.bcs.items():
        if bc.type == BoundaryConditionType.EMPTY:
            continue
        # Domain boundaries
        if isinstance(patch, DomainBoundaryPatch):
            mask = grid.get_domain_bnd_mask(patch)
            if not torch.any(mask):
                continue
            target_cells = grid.domain_bnd_owner[mask]
            Cf_bnd = grid.domain_bnd_face_centers[mask]
            C_O = grid.cell_centers[target_cells]
            mag_d = torch.linalg.vector_norm(Cf_bnd - C_O, dim=1, keepdim=True)
            yield BoundaryBatch(
                patch_name=patch,
                side=FaceSide.UPPER,
                face_kind=BoundaryFaceKind.DOMAIN,
                face_mask=mask,
                target_cells=target_cells,
                mag_d=mag_d,
            )
            continue

        # Immersed boundaries
        if isinstance(grid, AxisProjectedGrid):
            upper_mask, lower_mask = grid.ap_get_patch_mask(patch)
            # upper side
            if torch.any(upper_mask):
                yield BoundaryBatch(
                    patch_name=patch,
                    side=FaceSide.UPPER,
                    face_kind=BoundaryFaceKind.IMMERSED_UPPER,
                    face_mask=upper_mask,
                    target_cells=grid.owner[grid.ap_is_immersed_faces][
                        upper_mask
                    ],
                    mag_d=grid.ap_dist_owner_to_bnd[upper_mask],
                )

            if torch.any(lower_mask):
                yield BoundaryBatch(
                    patch_name=patch,
                    side=FaceSide.LOWER,
                    face_kind=BoundaryFaceKind.IMMERSED_LOWER,
                    face_mask=lower_mask,
                    target_cells=grid.neighbour[grid.ap_is_immersed_faces][
                        lower_mask
                    ],
                    mag_d=grid.ap_dist_neighbour_to_bnd[lower_mask],
                )
            continue


def evaluate_boundary_state(
    field: CellField, batch: BoundaryBatch
) -> tuple[
    Float[torch.Tensor, " F_any 1"],
    Float[torch.Tensor, " F_any k"],
    Float[torch.Tensor, " F_any k"],
    Float[torch.Tensor, " F_any k"],
]:
    bc = field.bcs[batch.patch_name]
    fraction, ref_v, ref_g = bc.evaluate(
        field, batch.patch_name, side=batch.side
    )
    psi_O = field.data[batch.target_cells]
    psi_b = fraction * ref_v + (1.0 - fraction) * (psi_O + ref_g * batch.mag_d)
    return fraction, ref_v, ref_g, psi_b
