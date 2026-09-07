"""
Static face geometry for octree grids, cached on ``grid.fv_cache``.

Depends only on mesh topology and the immersed-face set. On an axis-aligned
octree, skewness appears only at hanging-node (2:1) faces; those faces are
indexed separately so corrections run on that subset alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int

from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase


@dataclass(frozen=True)
class FaceGeometry:
    """
    Cached face geometry for one grid state.

    Attributes
    ----------
    owner, neighbour : torch.Tensor
        Owner / neighbour cells of all internal faces, ``[F]``.
    w_all : torch.Tensor
        Linear weights on all internal faces, ``[F]``
        (``psi_f = w * psi_O + (1 - w) * psi_N``).
    mag_Sf_all, delta_coeffs_all : torch.Tensor
        ``|Sf|`` and ``1 / |d · n|`` on all internal faces.
    single_mask, single_idx : torch.Tensor
        Single-sided (non-immersed) internal faces: bool mask ``[F]`` and
        integer indices.
    owner_s, neighbour_s, w_s, Sf_s, mag_Sf_s, delta_coeffs_s : torch.Tensor
        Single-sided subsets of the quantities above.
    hang_idx, hang_offset : torch.Tensor
        Hanging faces in the single-sided list and
        ``C_f - (w C_O + (1 - w) C_N)``.
    hang_move_O, hang_move_N : torch.Tensor
        In-face offsets from owner / neighbour centre to ``C_f``.
    hang_cell_mask, lsq_face_idx : torch.Tensor
        Cells next to hanging faces, and single-sided faces touching them.
    lsq_ata_inv : torch.Tensor
        Geometric least-squares normal-matrix inverse, ``[C, 3, 3]``.
    """

    owner: Int[torch.Tensor, " F"]
    neighbour: Int[torch.Tensor, " F"]
    w_all: Float[torch.Tensor, " F"]
    mag_Sf_all: Float[torch.Tensor, " F"]
    delta_coeffs_all: Float[torch.Tensor, " F"]

    single_mask: Bool[torch.Tensor, " F"]
    single_idx: Int[torch.Tensor, " F_single"]
    owner_s: Int[torch.Tensor, " F_single"]
    neighbour_s: Int[torch.Tensor, " F_single"]
    w_s: Float[torch.Tensor, " F_single"]
    Sf_s: Float[torch.Tensor, " F_single 3"]
    mag_Sf_s: Float[torch.Tensor, " F_single"]
    delta_coeffs_s: Float[torch.Tensor, " F_single"]

    hang_idx: Int[torch.Tensor, " F_hang"]
    hang_offset: Float[torch.Tensor, " F_hang 3"]
    hang_move_O: Float[torch.Tensor, " F_hang 3"]
    hang_move_N: Float[torch.Tensor, " F_hang 3"]
    hang_cell_mask: Bool[torch.Tensor, " C"]
    lsq_face_idx: Int[torch.Tensor, " F_lsq"]

    lsq_ata_inv: Float[torch.Tensor, " C 3 3"]

    @property
    def num_single(self) -> int:
        return int(self.single_idx.shape[0])

    @property
    def num_hanging(self) -> int:
        return int(self.hang_idx.shape[0])


def face_geometry(grid: IGridBase) -> FaceGeometry:
    """
    Return the cached :class:`FaceGeometry` for ``grid``.

    Stored on ``grid.fv_cache``; cleared by
    :meth:`~gridfoam.core.grid.base.IGridBase.invalidate_derived_caches`
    after ``remesh``, ``update_ib``, or ``to``.

    Parameters
    ----------
    grid : IGridBase
        Grid providing topology and geometry.

    Returns
    -------
    FaceGeometry
        Geometry for the current grid state.
    """
    cache = grid.fv_cache
    if cache.face_geometry is None:
        cache.face_geometry = _build_face_geometry(grid)
    return cache.face_geometry


def _build_face_geometry(grid: IGridBase) -> FaceGeometry:
    device = grid.device
    dtype = grid.dtype
    owner = grid.owner
    neighbour = grid.neighbour
    axis = grid.axis
    face_indices = torch.arange(grid.num_internal_faces, device=device)
    centers = grid.cell_centers

    # Axis-aligned linear weights and delta coefficients on all faces.
    d_all = centers[neighbour] - centers[owner]
    d_fN = centers[neighbour] - grid.face_centers
    # Select one normal component per face: (F, 3) -> (F,).
    d_n = torch.abs(d_all[face_indices, axis])
    w_all = torch.abs(d_fN[face_indices, axis]) / d_n
    mag_Sf_all = torch.linalg.vector_norm(grid.Sf, dim=1)
    delta_coeffs_all = 1.0 / d_n

    # Single-sided subset.
    single_mask = torch.ones(
        grid.num_internal_faces, dtype=torch.bool, device=device
    )
    if isinstance(grid, AxisProjectedGrid):
        single_mask[grid.ap_is_immersed_faces] = False
    single_idx = torch.nonzero(single_mask, as_tuple=True)[0]
    owner_s = owner[single_idx]
    neighbour_s = neighbour[single_idx]
    w_s = w_all[single_idx]
    Sf_s = grid.Sf[single_idx]
    mag_Sf_s = mag_Sf_all[single_idx]
    delta_coeffs_s = delta_coeffs_all[single_idx]
    axis_s = axis[single_idx]
    face_centers_s = grid.face_centers[single_idx]

    # Hanging-node faces: face centre off the owner--neighbour segment.
    centroid = (
        w_s[:, None] * centers[owner_s]
        + (1.0 - w_s)[:, None] * centers[neighbour_s]
    )
    offset = face_centers_s - centroid
    tol = 1.0e-8 * float(grid.cell_sizes.min().item())
    hang_mask = torch.linalg.vector_norm(offset, dim=1) > tol
    hang_idx = torch.nonzero(hang_mask, as_tuple=True)[0]
    hang_offset = offset[hang_idx]

    move_O = face_centers_s[hang_idx] - centers[owner_s[hang_idx]]
    move_N = face_centers_s[hang_idx] - centers[neighbour_s[hang_idx]]
    move_O.scatter_(1, axis_s[hang_idx, None], 0.0)
    move_N.scatter_(1, axis_s[hang_idx, None], 0.0)

    hang_cell_mask = torch.zeros(
        grid.num_cells, dtype=torch.bool, device=device
    )
    hang_cell_mask[owner_s[hang_idx]] = True
    hang_cell_mask[neighbour_s[hang_idx]] = True
    lsq_face_mask = hang_cell_mask[owner_s] | hang_cell_mask[neighbour_s]
    lsq_face_idx = torch.nonzero(lsq_face_mask, as_tuple=True)[0]

    lsq_ata_inv = _build_lsq_ata_inv(grid, owner_s, neighbour_s, dtype)

    return FaceGeometry(
        owner=owner,
        neighbour=neighbour,
        w_all=w_all,
        mag_Sf_all=mag_Sf_all,
        delta_coeffs_all=delta_coeffs_all,
        single_mask=single_mask,
        single_idx=single_idx,
        owner_s=owner_s,
        neighbour_s=neighbour_s,
        w_s=w_s,
        Sf_s=Sf_s,
        mag_Sf_s=mag_Sf_s,
        delta_coeffs_s=delta_coeffs_s,
        hang_idx=hang_idx,
        hang_offset=hang_offset,
        hang_move_O=move_O,
        hang_move_N=move_N,
        hang_cell_mask=hang_cell_mask,
        lsq_face_idx=lsq_face_idx,
        lsq_ata_inv=lsq_ata_inv,
    )


def _lsq_outer(d: Float[torch.Tensor, " N 3"]) -> Float[torch.Tensor, " N 3 3"]:
    """Inverse-distance-squared weighted outer product ``w2 d d^T``."""
    distance2 = torch.sum(d * d, dim=1)
    # A boundary through the center supplies a value constraint, not a slope.
    w2 = 1.0 / torch.where(distance2 > 0, distance2, torch.ones_like(distance2))
    # (N, 1, 1) * (N, 3, 1) * (N, 1, 3) -> (N, 3, 3).
    return w2[:, None, None] * d[:, :, None] * d[:, None, :]


def boundary_lsq_vectors(
    grid: IGridBase,
) -> list[tuple[Int[torch.Tensor, " N"], Float[torch.Tensor, " N 3"]]]:
    """
    Cell-to-boundary-face vectors for every boundary face of the grid.

    Domain-boundary faces use the true face centre; immersed faces use the
    axis-projected distance to the immersed surface on each side.

    Parameters
    ----------
    grid : IGridBase
        Grid providing boundary topology.

    Returns
    -------
    list of (cells, d)
        Target cells and displacement vectors for each boundary block.
    """
    blocks: list[tuple[torch.Tensor, torch.Tensor]] = []
    d_domain = (
        grid.domain_bnd_face_centers - grid.cell_centers[grid.domain_bnd_owner]
    )
    blocks.append((grid.domain_bnd_owner, d_domain))

    if isinstance(grid, AxisProjectedGrid) and grid.num_immersed_faces > 0:
        immersed = grid.ap_is_immersed_faces
        Sf = grid.Sf[immersed]
        mag_Sf = torch.linalg.vector_norm(Sf, dim=1)
        n_hat = Sf / mag_Sf[:, None]
        blocks.append(
            (
                grid.owner[immersed],
                grid.ap_dist_owner_to_bnd[:, None] * n_hat,
            )
        )
        blocks.append(
            (
                grid.neighbour[immersed],
                grid.ap_dist_neighbour_to_bnd[:, None] * (-n_hat),
            )
        )
    return blocks


def _build_lsq_ata_inv(
    grid: IGridBase,
    owner_s: Int[torch.Tensor, " F_single"],
    neighbour_s: Int[torch.Tensor, " F_single"],
    dtype: torch.dtype,
) -> Float[torch.Tensor, " C 3 3"]:
    centers = grid.cell_centers
    ata = torch.zeros((grid.num_cells, 3, 3), dtype=dtype, device=grid.device)
    d = centers[neighbour_s] - centers[owner_s]
    ata_face = _lsq_outer(d)
    ata.index_add_(0, owner_s, ata_face)
    ata.index_add_(0, neighbour_s, ata_face)

    for cells, d_bnd in boundary_lsq_vectors(grid):
        ata.index_add_(0, cells, _lsq_outer(d_bnd))

    # Coincident boundary points can leave a direction undetermined.
    return torch.linalg.pinv(ata, hermitian=True)
