"""Low-level kernels for interpolation onto internal faces."""

from __future__ import annotations

import torch
from jaxtyping import Bool, Float, Int

from gridfoam.core.field import CellField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase


def single_internal_mask(
    grid: IGridBase,
) -> Bool[torch.Tensor, " F_internal"]:
    """
    Boolean mask selecting single-sided internal faces.

    Immersed (double-sided) internal faces are excluded for
    axis-projected grids.

    Parameters
    ----------
    grid : IGridBase
        Grid providing face connectivity.

    Returns
    -------
    torch.Tensor
        Boolean mask over internal faces with shape ``[F_internal]``.
    """
    single_mask = torch.ones(
        grid.num_internal_faces, dtype=torch.bool, device=grid.device
    )
    if isinstance(grid, AxisProjectedGrid):
        single_mask[grid.ap_is_immersed_faces] = False
    return single_mask


def linear_face_weights(
    grid: IGridBase,
    face_mask: Bool[torch.Tensor, " F_internal"],
) -> tuple[
    Int[torch.Tensor, " F_sel"],
    Int[torch.Tensor, " F_sel"],
    Float[torch.Tensor, " F_sel 1"],
]:
    """
    Axis-aligned linear interpolation weights on selected internal faces.

    The interpolated face value is ``psi_f = w * psi_O + (1 - w) * psi_N``.

    Parameters
    ----------
    grid : IGridBase
        Grid providing cell centers and face centers.
    face_mask : torch.Tensor
        Boolean mask over internal faces.

    Returns
    -------
    tuple
        ``(owner, neighbour, w)`` indexed by the selected face subset.
    """
    owner = grid.owner[face_mask]
    neighbour = grid.neighbour[face_mask]
    axis_idx = grid.axis[face_mask, None]
    d_ON_vec = grid.cell_centers[neighbour] - grid.cell_centers[owner]
    d_fN_vec = grid.cell_centers[neighbour] - grid.face_centers[face_mask]
    d_ON = torch.abs(d_ON_vec.gather(1, axis_idx))
    d_fN = torch.abs(d_fN_vec.gather(1, axis_idx))
    w = d_fN / d_ON
    return owner, neighbour, w


def linear_internal_face_values(
    field: CellField,
) -> Float[torch.Tensor, " F_single k"]:
    """
    Uncorrected axis-aligned linear values on single-sided internal faces.

    Parameters
    ----------
    field : CellField
        Cell-centered field.

    Returns
    -------
    torch.Tensor
        Face values with shape ``[F_single, k]``.
    """
    grid = field.grid
    single_mask = single_internal_mask(grid)
    owner, neighbour, w = linear_face_weights(grid, single_mask)
    return w * field.data[owner] + (1.0 - w) * field.data[neighbour]


def correct_internal_values(
    field: CellField,
    base_values: Float[torch.Tensor, " F_single k"],
    grad_data: Float[torch.Tensor, " C k 3"] | Float[torch.Tensor, " C _"],
) -> Float[torch.Tensor, " F_single k"]:
    """
    Add face-centre offset correction to base linear face values.

    ``psi_f = base + grad_f & (C_f - C_w)``,

    where ``C_w`` is the weighted cell-centre interpolation point and
    ``grad_f`` is the same weighted interpolation of the cell gradient.

    Parameters
    ----------
    field : CellField
        Cell-centered field.
    base_values : torch.Tensor
        Uncorrected linear face values with shape ``[F_single, k]``.
    grad_data : torch.Tensor
        Cell-centered gradient with shape ``[C, k, 3]`` or flat
        ``[C, k * 3]``.

    Returns
    -------
    torch.Tensor
        Corrected face values with shape ``[F_single, k]``.
    """
    grid = field.grid
    k = field.num_components
    single_mask = single_internal_mask(grid)
    owner, neighbour, w = linear_face_weights(grid, single_mask)

    if grad_data.ndim == 2:
        grad_tensor = grad_data.reshape(grid.num_cells, k, 3)
    else:
        grad_tensor = grad_data

    grad_o = grad_tensor[owner]
    grad_n = grad_tensor[neighbour]
    grad_face = w[:, None] * grad_o + (1.0 - w)[:, None] * grad_n

    centroid = (
        w * grid.cell_centers[owner] + (1.0 - w) * grid.cell_centers[neighbour]
    )
    face_offset = grid.face_centers[single_mask] - centroid
    correction = torch.sum(grad_face * face_offset[:, None, :], dim=2)
    return base_values + correction
