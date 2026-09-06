"""Low-level kernels for interpolation onto internal faces."""

from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField
from gridfoam.fv.kernels.face_geometry import FaceGeometry, face_geometry


def linear_internal_face_values(
    field: CellField,
    geometry: FaceGeometry | None = None,
) -> Float[torch.Tensor, " F_single k"]:
    """
    Uncorrected axis-aligned linear values on single-sided internal faces.

    Parameters
    ----------
    field : CellField
        Cell-centered field.
    geometry : FaceGeometry or None
        Cached face geometry. Looked up from the grid when ``None``.

    Returns
    -------
    torch.Tensor
        Face values with shape ``[F_single, k]``.
    """
    geo = geometry if geometry is not None else face_geometry(field.grid)
    psi = field.data
    return geo.w_s * psi[geo.owner_s] + (1.0 - geo.w_s) * psi[geo.neighbour_s]


def correct_internal_values(
    field: CellField,
    base_values: Float[torch.Tensor, " F_single k"],
    grad_data: Float[torch.Tensor, " C k 3"] | Float[torch.Tensor, " C _"],
    geometry: FaceGeometry | None = None,
) -> Float[torch.Tensor, " F_single k"]:
    """
    Add face-centre offset correction to base linear face values.

    ``psi_f = base + grad_f & (C_f - C_w)``,

    where ``C_w`` is the weighted cell-centre interpolation point and
    ``grad_f`` is the same weighted interpolation of the cell gradient.
    The offset vanishes on regular faces, so the correction is evaluated
    only on hanging-node faces; ``grad_data`` needs to be valid only on the
    cells adjacent to those faces.

    Parameters
    ----------
    field : CellField
        Cell-centered field.
    base_values : torch.Tensor
        Uncorrected linear face values with shape ``[F_single, k]``.
    grad_data : torch.Tensor
        Cell-centered gradient with shape ``[C, k, 3]`` or flat
        ``[C, k * 3]``.
    geometry : FaceGeometry or None
        Cached face geometry. Looked up from the grid when ``None``.

    Returns
    -------
    torch.Tensor
        Corrected face values with shape ``[F_single, k]``.
    """
    grid = field.grid
    geo = geometry if geometry is not None else face_geometry(grid)
    if geo.num_hanging == 0:
        return base_values

    k = field.num_components
    if grad_data.ndim == 2:
        grad_tensor = grad_data.reshape(grid.num_cells, k, 3)
    else:
        grad_tensor = grad_data

    hang = geo.hang_idx
    owner = geo.owner_s[hang]
    neighbour = geo.neighbour_s[hang]
    w = geo.w_s[hang]
    grad_face = (
        w[:, None] * grad_tensor[owner]
        + (1.0 - w)[:, None] * grad_tensor[neighbour]
    )
    correction = torch.sum(grad_face * geo.hang_offset[:, None, :], dim=2)

    corrected = base_values.clone()
    corrected.index_add_(0, hang, correction)
    return corrected


def sn_grad_hanging_correction(
    field: CellField,
    grad_data: Float[torch.Tensor, " C k 3"],
    geometry: FaceGeometry | None = None,
) -> Float[torch.Tensor, " F_hang k"]:
    """
    Skewness correction of the surface-normal gradient on hanging faces.

    Each cell value is reconstructed onto the face-normal line through the
    face centre using the cell gradient, removing the tangential offset of
    the cell centre from that line. The correction to the compact
    difference ``(psi_N - psi_O) / |d . n|`` is

    ``(grad_N & t_N - grad_O & t_O) / |d . n|``,

    with ``t_O = (C_f - C_O)`` and ``t_N = (C_f - C_N)`` projected onto the
    face plane. It is exact for linear fields whenever the supplied
    gradient is, and is zero on regular faces.

    Parameters
    ----------
    field : CellField
        Cell-centered field.
    grad_data : torch.Tensor
        Cell gradient ``[C, k, 3]`` valid on hanging-adjacent cells.
    geometry : FaceGeometry or None
        Cached face geometry. Looked up from the grid when ``None``.

    Returns
    -------
    torch.Tensor
        Normal-gradient correction on hanging faces, ``[F_hang, k]``.
    """
    geo = geometry if geometry is not None else face_geometry(field.grid)
    hang = geo.hang_idx
    owner = geo.owner_s[hang]
    neighbour = geo.neighbour_s[hang]
    psi_corr = torch.sum(
        grad_data[neighbour] * geo.hang_move_N[:, None, :], dim=2
    ) - torch.sum(grad_data[owner] * geo.hang_move_O[:, None, :], dim=2)
    return psi_corr * geo.delta_coeffs_s[hang]
