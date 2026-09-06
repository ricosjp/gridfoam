from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum, auto

import torch
from jaxtyping import Bool, Float, Int

from gridfoam.core.field import CellField, FaceField
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
    """
    One patch/side block of boundary faces sharing a boundary condition.

    Attributes
    ----------
    patch_name : PatchName
        Patch the boundary condition is attached to.
    side : FaceSide
        Upper/lower side (only meaningful for immersed faces).
    face_kind : BoundaryFaceKind
        Storage block the faces live in.
    face_mask : torch.Tensor
        Boolean mask into that storage block.
    target_cells : torch.Tensor
        Adjacent cell for each selected face.
    d_vec : torch.Tensor
        Vector from the cell centre to the boundary point, ``[F_any, 3]``.
    mag_d : torch.Tensor
        ``|d_vec|`` with shape ``[F_any, 1]``.
    """

    patch_name: PatchName
    side: FaceSide
    face_kind: BoundaryFaceKind
    face_mask: Bool[torch.Tensor, " F_any"]
    target_cells: Int[torch.Tensor, " F_any"]
    d_vec: Float[torch.Tensor, " F_any 3"]
    mag_d: Float[torch.Tensor, " F_any 1"]


BoundaryBatchCacheKey = tuple[
    tuple[int, int, int, int, int, tuple[float, ...], tuple[float, ...]]
    | tuple[int, int],
    tuple[tuple[PatchName, int, BoundaryConditionType], ...],
]


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
    """
    Yield boundary-face batches for ``field``.

    Results are cached on ``field.fv_cache`` and rebuilt when topology or
    the boundary-condition set changes.
    """
    cache = field.fv_cache
    cache_key = _boundary_batch_cache_key(field)
    if (
        cache.boundary_batches is not None
        and cache.boundary_batches_key == cache_key
    ):
        yield from cache.boundary_batches
        return

    batches = tuple(_build_boundary_batches(field))
    cache.boundary_batches_key = cache_key
    cache.boundary_batches = batches
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
            d_vec = Cf_bnd - C_O
            mag_d = torch.linalg.vector_norm(d_vec, dim=1, keepdim=True)
            yield BoundaryBatch(
                patch_name=patch,
                side=FaceSide.UPPER,
                face_kind=BoundaryFaceKind.DOMAIN,
                face_mask=mask,
                target_cells=target_cells,
                d_vec=d_vec,
                mag_d=mag_d,
            )
            continue

        # Immersed boundaries
        if isinstance(grid, AxisProjectedGrid):
            upper_mask, lower_mask = grid.ap_get_patch_mask(patch)
            immersed_Sf = grid.Sf[grid.ap_is_immersed_faces]
            n_hat = immersed_Sf / torch.linalg.vector_norm(
                immersed_Sf, dim=1, keepdim=True
            )
            # upper side
            if torch.any(upper_mask):
                mag_d = grid.ap_dist_owner_to_bnd[upper_mask]
                yield BoundaryBatch(
                    patch_name=patch,
                    side=FaceSide.UPPER,
                    face_kind=BoundaryFaceKind.IMMERSED_UPPER,
                    face_mask=upper_mask,
                    target_cells=grid.owner[grid.ap_is_immersed_faces][
                        upper_mask
                    ],
                    d_vec=n_hat[upper_mask] * mag_d,
                    mag_d=mag_d,
                )

            if torch.any(lower_mask):
                mag_d = grid.ap_dist_neighbour_to_bnd[lower_mask]
                yield BoundaryBatch(
                    patch_name=patch,
                    side=FaceSide.LOWER,
                    face_kind=BoundaryFaceKind.IMMERSED_LOWER,
                    face_mask=lower_mask,
                    target_cells=grid.neighbour[grid.ap_is_immersed_faces][
                        lower_mask
                    ],
                    d_vec=-n_hat[lower_mask] * mag_d,
                    mag_d=mag_d,
                )
            continue


def boundary_block(
    face_field: FaceField, face_kind: BoundaryFaceKind
) -> Float[torch.Tensor, " F_any k"]:
    """
    Storage block of ``face_field`` for a boundary face kind.

    Parameters
    ----------
    face_field : FaceField
        Face field to index.
    face_kind : BoundaryFaceKind
        Domain, immersed-upper or immersed-lower block.

    Returns
    -------
    torch.Tensor
        The block tensor (a view; in-place writes update the field).
    """
    if face_kind == BoundaryFaceKind.DOMAIN:
        return face_field.domain_bnd_data
    if face_kind == BoundaryFaceKind.IMMERSED_UPPER:
        return face_field.immersed_upper
    return face_field.immersed_lower


def outward_boundary_Sf(
    grid: IGridBase, batch: BoundaryBatch
) -> Float[torch.Tensor, " F_any 3"]:
    """
    Outward face-area vectors of the faces selected by ``batch``.

    Parameters
    ----------
    grid : IGridBase
        Grid providing face geometry.
    batch : BoundaryBatch
        Boundary face block.

    Returns
    -------
    torch.Tensor
        Outward ``Sf`` with shape ``[F_any, 3]``.
    """
    if batch.face_kind == BoundaryFaceKind.DOMAIN:
        return grid.domain_bnd_Sf[batch.face_mask]
    assert isinstance(grid, AxisProjectedGrid)
    immersed_Sf = grid.Sf[grid.ap_is_immersed_faces][batch.face_mask]
    if batch.face_kind == BoundaryFaceKind.IMMERSED_UPPER:
        return immersed_Sf
    return -immersed_Sf


def uncovered_domain_faces(field: CellField) -> Bool[torch.Tensor, " F_bnd"]:
    """
    Domain-boundary faces of ``field`` without a boundary condition.

    For a primary field such as ``U`` these are the ``empty`` (2-D) patches:
    no flux may cross them, whereas value extrapolation is still used for
    gradients and interpolation.

    Parameters
    ----------
    field : CellField
        Field whose boundary conditions are inspected.

    Returns
    -------
    torch.Tensor
        Boolean mask over ``grid.num_domain_bnd_faces``.
    """
    grid = field.grid
    covered = torch.zeros(
        grid.num_domain_bnd_faces, dtype=torch.bool, device=grid.device
    )
    for batch in iter_boundary_batches(field):
        if batch.face_kind == BoundaryFaceKind.DOMAIN:
            covered |= batch.face_mask
    return ~covered


@dataclass(frozen=True)
class BoundaryState:
    """
    Evaluated boundary condition of one :class:`BoundaryBatch`.

    Attributes
    ----------
    batch : BoundaryBatch
        Face block the state belongs to.
    fraction : torch.Tensor
        Dirichlet blend fraction, ``[F_any, 1]``.
    ref_v : torch.Tensor
        Reference value for the Dirichlet part, ``[F_any, k]``.
    ref_g : torch.Tensor
        Reference normal gradient for the Neumann part, ``[F_any, k]``.
    psi_b : torch.Tensor
        Resulting boundary face value, ``[F_any, k]``.
    """

    batch: BoundaryBatch
    fraction: Float[torch.Tensor, " F_any 1"]
    ref_v: Float[torch.Tensor, " F_any k"]
    ref_g: Float[torch.Tensor, " F_any k"]
    psi_b: Float[torch.Tensor, " F_any k"]


_BoundaryStateKey = tuple[
    BoundaryBatchCacheKey,
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
]


def _boundary_state_cache_key(field: CellField) -> _BoundaryStateKey:
    dependency_tokens: list[tuple[int, ...]] = []
    for bc in field.bcs.values():
        for dep in bc.dependencies(field):
            dependency_tokens.append((id(dep),) + dep.state_token())
    return (
        _boundary_batch_cache_key(field),
        field.state_token(),
        tuple(dependency_tokens),
    )


def _compute_boundary_state(
    field: CellField, batch: BoundaryBatch
) -> BoundaryState:
    bc = field.bcs[batch.patch_name]
    fraction, ref_v, ref_g = bc.evaluate(
        field, batch.patch_name, side=batch.side
    )
    psi_O = field.data[batch.target_cells]
    psi_b = fraction * ref_v + (1.0 - fraction) * (psi_O + ref_g * batch.mag_d)
    return BoundaryState(batch, fraction, ref_v, ref_g, psi_b)


def _boundary_states(field: CellField) -> dict[int, BoundaryState]:
    """
    Boundary states for every batch of ``field``, keyed by ``id(batch)``.

    Cached on ``field.fv_cache``; invalidated when field data, BCs, grid
    state, or BC dependency fields change.
    """
    cache = field.fv_cache
    key = _boundary_state_cache_key(field)
    if cache.boundary_states is not None and cache.boundary_states_key == key:
        return cache.boundary_states
    states = {
        id(batch): _compute_boundary_state(field, batch)
        for batch in iter_boundary_batches(field)
    }
    cache.boundary_states_key = key
    cache.boundary_states = states
    return states


def iter_boundary_states(field: CellField) -> Iterator[BoundaryState]:
    """
    Iterate over cached boundary states of ``field``.

    Parameters
    ----------
    field : CellField
        Field whose boundary conditions are evaluated.

    Yields
    ------
    BoundaryState
        One state per boundary batch, in ``iter_boundary_batches`` order.
    """
    states = _boundary_states(field)
    for batch in iter_boundary_batches(field):
        yield states[id(batch)]


def evaluate_boundary_state(
    field: CellField, batch: BoundaryBatch
) -> tuple[
    Float[torch.Tensor, " F_any 1"],
    Float[torch.Tensor, " F_any k"],
    Float[torch.Tensor, " F_any k"],
    Float[torch.Tensor, " F_any k"],
]:
    """
    Return ``(fraction, ref_v, ref_g, psi_b)`` for ``batch``.

    Served from the per-field boundary-state cache when ``batch`` comes
    from :func:`iter_boundary_batches`; otherwise evaluated directly.
    """
    state = _boundary_states(field).get(id(batch))
    if state is None or state.batch is not batch:
        state = _compute_boundary_state(field, batch)
    return state.fraction, state.ref_v, state.ref_g, state.psi_b


def fill_boundary_face_values(field: CellField, psi_f: FaceField) -> None:
    """
    Write boundary face values of ``field`` into ``psi_f``.

    Faces covered by a boundary condition receive the BC face value. Faces
    of patches without a boundary condition (derived fields such as
    ``grad(p)``, ``HbyA`` or ``rAU``, and ``empty`` patches) are filled by
    zero-gradient extrapolation from the adjacent cell, mirroring OpenFOAM's
    ``extrapolatedCalculated`` patch type. Leaving those faces at zero would
    pollute Green-Gauss sums and skew corrections near the boundary.

    Parameters
    ----------
    field : CellField
        Cell-centered field supplying values and boundary conditions.
    psi_f : FaceField
        Target face field whose boundary blocks are overwritten.
    """
    grid = field.grid
    device = grid.device
    n_immersed = (
        grid.num_immersed_faces if isinstance(grid, AxisProjectedGrid) else 0
    )
    covered_upper = torch.zeros(n_immersed, dtype=torch.bool, device=device)
    covered_lower = torch.zeros_like(covered_upper)

    for state in iter_boundary_states(field):
        batch = state.batch
        boundary_block(psi_f, batch.face_kind)[batch.face_mask] = state.psi_b
        if batch.face_kind == BoundaryFaceKind.IMMERSED_UPPER:
            covered_upper |= batch.face_mask
        elif batch.face_kind == BoundaryFaceKind.IMMERSED_LOWER:
            covered_lower |= batch.face_mask

    uncovered = uncovered_domain_faces(field)
    if bool(torch.any(uncovered)):
        cells = grid.domain_bnd_owner[uncovered]
        psi_f.domain_bnd_data[uncovered] = field.data[cells]

    if isinstance(grid, AxisProjectedGrid) and n_immersed > 0:
        immersed = grid.ap_is_immersed_faces
        uncovered_upper = ~covered_upper
        if bool(torch.any(uncovered_upper)):
            cells = grid.owner[immersed][uncovered_upper]
            psi_f.immersed_upper[uncovered_upper] = field.data[cells]
        uncovered_lower = ~covered_lower
        if bool(torch.any(uncovered_lower)):
            cells = grid.neighbour[immersed][uncovered_lower]
            psi_f.immersed_lower[uncovered_lower] = field.data[cells]
